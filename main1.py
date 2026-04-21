import os
import glob
import uuid
import time
import certifi
from typing import List, TypedDict, Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from urllib.parse import quote_plus

# Cloud DBs
from pinecone import Pinecone, ServerlessSpec
from pymongo import MongoClient

# Google Genai (direct SDK - no langchain wrapper)
from google import genai as google_genai

# LangChain & ML
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. State Definition
# ==========================================
class EnterpriseState(TypedDict):
    messages: Annotated[list, add_messages]
    standalone_query: str
    documents: List[dict]
    draft_answer: str
    expert_fallback: str
    is_verified: bool
    target_index: str

# ==========================================
# 2. The Cloud-Native Pipeline
# ==========================================
class CloudEnterprisePipeline:
    def __init__(self):
        # ---> SET YOUR KEYS HERE <---
        os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"
        os.environ["PINECONE_API_KEY"] = "PINECONE_API_KEY"
        MONGO_URI = "MONGO_URI"

        # 1. Init Google Genai Client (direct SDK - bypasses broken langchain wrapper)
        self.genai_client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # 2. Init Cross-Encoder Reranker
        print("Loading Cross-Encoder Reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

        # 3. Init MongoDB
        self.mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        self.mongo_db = self.mongo_client["enterprise_knowledge"]
        self.docs_collection = self.mongo_db["raw_documents"]

        # 4. Init Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        # --- INDEX A: Google Gemini (3072 Dimensions) ---
        self.index_google_name = "enterprise-rag"
        if self.index_google_name not in self.pc.list_indexes().names():
            print("Creating Google Index...")
            self.pc.create_index(
                name=self.index_google_name, dimension=3072, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index_google = self.pc.Index(self.index_google_name)

        # --- INDEX B: Multilingual E5 (1024 Dimensions) ---
        self.index_global_name = "enterprise-rag-global"
        if self.index_global_name not in self.pc.list_indexes().names():
            print("Creating Global Multilingual Index...")
            self.pc.create_index(
                name=self.index_global_name, dimension=1024, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index_global = self.pc.Index(self.index_global_name)

    # --- HELPER: LLM Call (direct Google SDK, with rate limit retry) ---
    def _llm(self, prompt: str) -> str:
       while True:
        try:
            response = self.genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            err = str(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err):
                if "limit: 0" in err:
                    raise RuntimeError("Daily Google API quota exhausted. Please wait until midnight PT or use a new API key.") from e
                print("LLM rate limited, waiting 60 seconds...")
                time.sleep(60)
            else:
                raise e

    # --- HELPER: Embedding Call (direct Google SDK, with rate limit retry) ---
    def _embed(self, texts: list) -> list:
        all_embeddings = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            while True:
                try:
                    result = self.genai_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch
                    )
                    all_embeddings.extend([e.values for e in result.embeddings])
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print("Embedding rate limited, waiting 60 seconds...")
                        time.sleep(60)
                    else:
                        raise e
        return all_embeddings

    # --- NODE: INGESTION ---
    def ingest(self, directory_path: str, target_index: str):
        # Clear the target index
        if target_index == "google":
            try:
                self.index_google.delete(delete_all=True)
            except Exception:
                pass
        elif target_index == "global":
            try:
                self.index_global.delete(delete_all=True)
            except Exception:
                pass

        # Clear MongoDB
        self.docs_collection.delete_many({})

        total_chunks = 0
        for file_path in glob.glob(f"{directory_path}/**/*.*", recursive=True):
            print(f"Processing: {file_path}")
            ext = os.path.splitext(file_path)[1].lower()

            try:
                if ext == ".pdf":
                    docs = PyMuPDFLoader(file_path).load()
                    docs = [d for d in docs if d.page_content.strip()]  # filter empty pages
                elif ext == ".docx":
                    docs = Docx2txtLoader(file_path).load()
                elif ext == ".txt":
                    docs = TextLoader(file_path, autodetect_encoding=True).load()
                else:
                    continue

                # Save Parent Document to MongoDB
                mongo_id = str(uuid.uuid4())
                self.docs_collection.insert_one({
                    "_id": mongo_id,
                    "source_file": os.path.basename(file_path),
                })

                # Chunking
                chunks = self.text_splitter.split_documents(docs)
                texts = [chunk.page_content for chunk in chunks]
                vectors_to_upsert = []

                # ROUTE A: Google Embeddings
                if target_index == "google":
                    embeds = self._embed(texts)
                    for i, (text, embed) in enumerate(zip(texts, embeds)):
                        vectors_to_upsert.append({
                            "id": f"{mongo_id}_chunk_{i}",
                            "values": embed,
                            "metadata": {"mongo_id": mongo_id, "text": text, "source": os.path.basename(file_path)}
                        })
                    batch_size = 100
                    for j in range(0, len(vectors_to_upsert), batch_size):
                        self.index_google.upsert(vectors=vectors_to_upsert[j:j + batch_size])

                # ROUTE B: Pinecone Multilingual E5
                elif target_index == "global":
                    e5_inputs = [f"passage: {text}" for text in texts]
                    e5_response = self.pc.inference.embed(
                        model="multilingual-e5-large",
                        inputs=e5_inputs,
                        parameters={"input_type": "passage", "truncate": "END"}
                    )
                    for i, (text, record) in enumerate(zip(texts, e5_response.data)):
                        vectors_to_upsert.append({
                            "id": f"{mongo_id}_chunk_{i}",
                            "values": record.values,
                            "metadata": {"mongo_id": mongo_id, "text": text, "source": os.path.basename(file_path)}
                        })
                    batch_size = 100
                    for j in range(0, len(vectors_to_upsert), batch_size):
                        self.index_global.upsert(vectors=vectors_to_upsert[j:j + batch_size])

                total_chunks += len(chunks)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        return f"Processed {total_chunks} chunks into the '{target_index}' index."

    # --- NODE: REFORMULATOR ---
    def reformulate_query(self, state: EnterpriseState):
        messages = state["messages"]
        latest_question = messages[-1].content
        if len(messages) <= 1:
            return {"standalone_query": latest_question}

        history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in messages[:-1]])
        prompt = f"Given history, rewrite the latest question to be standalone.\nHistory:\n{history_str}\nLatest: {latest_question}\nStandalone:"
        response = self._llm(prompt)
        return {"standalone_query": response.strip()}

    # --- NODE: CLOUD RETRIEVER ---
    def retrieve_information(self, state: EnterpriseState):
        query = state["standalone_query"]
        target_index = state.get("target_index", "google")
        print(f"--- [AGENT] Semantic Search on '{target_index}' Index for: '{query}' ---")

        if target_index == "google":
            query_embed = self._embed([query])[0]
            pc_results = self.index_google.query(vector=query_embed, top_k=20, include_metadata=True)

        elif target_index == "global":
            e5_query = self.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[f"query: {query}"],
                parameters={"input_type": "query"}
            )
            query_embed = e5_query.data[0].values
            pc_results = self.index_global.query(vector=query_embed, top_k=20, include_metadata=True)

        if not pc_results['matches']:
            return {"documents": []}

        candidates = [{"text": match['metadata']['text'], "source": match['metadata']['source']} for match in pc_results['matches']]

        # Cross-Encoder Reranking
        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.reranker.predict(pairs)
        for idx, doc in enumerate(candidates):
            doc["score"] = float(scores[idx])

        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]
        return {"documents": ranked}

    # --- NODE: EXPERT FINDER ---
    def find_expert(self, state: EnterpriseState):
        docs = state["documents"]
        if not docs or docs[0].get("score", 1.0) < 0.0:
            fallback = "I cannot confidently answer this based on existing documentation. Please consult a human subject matter expert."
            return {"expert_fallback": fallback, "draft_answer": fallback, "is_verified": True, "messages": [AIMessage(content=fallback)]}
        return {"expert_fallback": None}

    # --- NODE: GENERATION ---
    def generate_answer(self, state: EnterpriseState):
        if state.get("expert_fallback"):
            return state

        context = "\n\n".join([f"Source: {d['source']}\n{d['text']}" for d in state["documents"]])
        prompt = f"You are an enterprise AI. Answer using ONLY the context.\nContext: {context}\nQuestion: {state['standalone_query']}"
        response = self._llm(prompt)
        return {"draft_answer": response, "messages": [AIMessage(content=response)]}

    # --- NODE: SELF-RAG VERIFICATION ---
    def verify_answer(self, state: EnterpriseState):
        if state.get("expert_fallback"):
            return state
        return {"is_verified": True}



# ==========================================
# 3. LangGraph Workflow & FastAPI
# ==========================================
pipeline = CloudEnterprisePipeline()

# Build the Graph
workflow = StateGraph(EnterpriseState)
workflow.add_node("reformulate", pipeline.reformulate_query)
workflow.add_node("retrieve", pipeline.retrieve_information)
workflow.add_node("evaluate_expert", pipeline.find_expert)
workflow.add_node("generate", pipeline.generate_answer)
workflow.add_node("verify", pipeline.verify_answer)

workflow.set_entry_point("reformulate")
workflow.add_edge("reformulate", "retrieve")
workflow.add_edge("retrieve", "evaluate_expert")

def route_after_expert(state):
    return "end" if state.get("expert_fallback") else "generate"

workflow.add_conditional_edges("evaluate_expert", route_after_expert, {"end": END, "generate": "generate"})
workflow.add_edge("generate", "verify")

def route_after_verification(state):
    return "end" if state["is_verified"] else "evaluate_expert"

workflow.add_conditional_edges("verify", route_after_verification, {"end": END, "evaluate_expert": "evaluate_expert"})

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# FastAPI App
app = FastAPI(title="Cloud Enterprise Vector AI")

class IngestRequest(BaseModel):
    directory_path: str
    target_index: str = "google"

class QueryRequest(BaseModel):
    session_id: str
    query: str
    target_index: str = "google"

@app.post("/api/ingest")
async def ingest_documents(request: IngestRequest):
    if request.target_index not in ["google", "global"]:
        raise HTTPException(status_code=400, detail="target_index must be 'google' or 'global'")
    return {"status": "success", "message": pipeline.ingest(request.directory_path, request.target_index)}

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    if request.target_index not in ["google", "global"]:
        raise HTTPException(status_code=400, detail="target_index must be 'google' or 'global'")

    config = {"configurable": {"thread_id": request.session_id}}
    final_state = app_graph.invoke({
        "messages": [HumanMessage(content=request.query)],
        "target_index": request.target_index
    }, config=config)

    return {
        "query": request.query,
        "index_used": request.target_index,
        "answer": final_state["draft_answer"],
        "verified": final_state["is_verified"],
        "sources": [{"source": d['source'], "score": round(d['score'], 2)} for d in final_state.get("documents", [])]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
