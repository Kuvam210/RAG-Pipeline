# 🚀 Cloud Enterprise Vector AI (RAG Pipeline)

This project is a **cloud-native Retrieval-Augmented Generation (RAG) system** built using FastAPI, Pinecone, MongoDB, Google GenAI, and LangGraph.

It allows you to:

* Ingest documents (PDF, DOCX, TXT)
* Store embeddings in vector databases
* Perform semantic search
* Generate context-aware AI responses
* Maintain conversational memory

---

## 🧠 Architecture Overview

* **LLM**: Google Gemini (via direct SDK)
* **Vector DB**: Pinecone (dual index system)
* **Database**: MongoDB (document storage)
* **Reranking**: Cross-Encoder (MS MARCO)
* **Framework**: LangGraph (stateful workflow)
* **Backend API**: FastAPI

---

## ⚙️ Features

* 🔍 Semantic search with reranking
* 🌍 Multilingual + standard embedding pipelines
* 🧾 Document ingestion (PDF, DOCX, TXT)
* 🧠 Conversational memory (LangGraph)
* 🧪 Self-verification step for answers
* 🧑‍💼 Expert fallback when confidence is low

---

## 📁 Project Structure

```
main1.py        # Core pipeline + FastAPI app
```

---

## 🔑 Environment Variables

**⚠️ IMPORTANT: Never hardcode API keys in production**

Set the following:

```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
MONGO_URI=your_mongodb_connection_string
```

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <repo-name>

pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
python main1.py
```

Server will start at:

```
http://0.0.0.0:8000
```

---

## 📡 API Endpoints

### 1. Ingest Documents

**POST** `/api/ingest`

```json
{
  "directory_path": "path/to/docs",
  "target_index": "google"
}
```

* `target_index`: `"google"` or `"global"`

---

### 2. Ask Question

**POST** `/api/ask`

```json
{
  "session_id": "user123",
  "query": "Your question here",
  "target_index": "google"
}
```

---

## 🔄 Workflow Pipeline

1. Reformulate query
2. Retrieve relevant chunks
3. Rerank results
4. Generate answer
5. Verify response
6. Fallback to expert if needed

---

## 🧪 Supported File Types

* `.pdf`
* `.docx`
* `.txt`

---

## ⚠️ Known Issues / Warnings

* API keys are hardcoded in code (fix this immediately)
* Rate limits from Google API may slow responses
* Pinecone index creation may take time
* MongoDB Atlas connection required

---

## 🧠 Future Improvements

* Add authentication
* Add frontend UI
* Improve verification logic
* Add streaming responses
* Dockerize the app

---

## 📜 License

Add your preferred license here (MIT recommended)
