# 🦙 Deep Dive into LlamaIndex

A thorough, self‑contained introduction to **LlamaIndex** – the open‑source framework that helps you build **LLM applications** powered by your own data.  
This repository captures the core concepts, architecture, and code from the introductory video, turning it into a permanent reference you can revisit anytime.

> **Source video:** [LlamaIndex Introduction (Part 1)](https://youtu.be/cCyYGYyCka4)

---

## 📌 What is LlamaIndex?

LlamaIndex is a **data framework for LLM applications**.  
It provides:

- **Data connectors** to ingest unstructured/structured data (PDF, HTML, CSV, Word, Notion, …)
- **Document & node abstractions** that break down your data into a rich, interconnected knowledge graph
- **Embedding + indexing** to store and retrieve information efficiently
- **Retrieval & query engines** that answer questions over your custom data
- **Enterprise‑grade cloud services** (LlamaCloud, LlamaParse) for production‑ready ingestion

In short, LlamaIndex **enriches the knowledge of any large language model with your private or domain‑specific data**.

### 🔄 LlamaIndex vs LangChain

| LlamaIndex | LangChain |
|------------|-----------|
| Specialises in **data indexing & retrieval** | General‑purpose LLM application framework |
| Uses **nodes** as first‑class citizens | Often works with raw document chunks |
| Built‑in support for **routers & retrievers** | More modular – you assemble these yourself |
| LlamaParse for complex document parsing | No equivalent built‑in parsing service |

Both can be used together, and LlamaIndex can integrate LangChain components seamlessly.

---

## 🧱 Core Architecture (RAG Pipeline)
<img width="2646" height="1460" alt="Gemini_Generated_Image_gu7pargu7pargu7p" src="https://github.com/user-attachments/assets/5d1071ec-6b5e-41b2-8aed-9dcf404e6d9d" />

Below is a textual representation of the LlamaIndex data processing and retrieval pipeline.  
Each coloured box is a core component of the framework.
```
Unstructured / Structured Data
        |
        v
[ Data Connectors ]   ← LlamaHub, SimpleDirectoryReader, LlamaParse
        |
        v
[ Documents ]         ← metadata + raw text
        |
        v
[ Nodes ]              ← interconnected, granular chunks
        |
        v
[ Embeddings ]         ← numerical vectors
        |
        v
[ Index ]              ← Vector Store Index (e.g. in‑memory, Chroma, Pinecone)
        |
        v
[ Router → Retrievers ]  ← multi‑strategy retrieval
        |
        v
[ Response Synthesizer ]  ← prompt + context → LLM → answer
```

### Component Breakdown

1. **Data Connectors**  
   - Load data from PDF, HTML, CSV, Word, databases, Notion, etc.  
   - `SimpleDirectoryReader` can automatically handle most file types.  
   - **LlamaHub** provides community‑contributed connectors.  
   - **LlamaParse** (cloud API) returns clean, structured text even from complex documents with tables and images.

2. **Documents**  
   - A programming object containing:  
     - `text`: the raw text from the source  
     - `metadata`: file name, page number, source path, dates, etc.  
   - Documents are the output of data connectors.

3. **Nodes**  
   - Granular pieces of a document (e.g. a chunk).  
   - **Inherit metadata** from the parent document.  
   - **Interconnected** across all documents, forming a knowledge graph.  
   - LlamaIndex explicitly works with nodes, making the ingestion pipeline very transparent.

4. **Embeddings**  
   - Each node is converted into a numerical vector using an embedding model (OpenAI, local, Hugging Face, …).  
   - Vectors capture semantic meaning.

5. **Index**  
   - A **vector store** that holds all embeddings + original text.  
   - `VectorStoreIndex` is the most common one; can be backed by in‑memory, ChromaDB, Pinecone, etc.  
   - This is what you later query to retrieve relevant information.

6. **Router & Retrievers**  
   - **Router** decides which retrieval strategy to use for a given query.  
   - **Retrievers** implement different search methods (top‑k similarity, keyword, hybrid).  
   - They pull the most relevant nodes from the index.

7. **Response Synthesizer**  
   - Takes the retrieved nodes, combines them with the user’s query, and prompts the LLM.  
   - Returns a coherent, context‑aware answer.

---

## 🚀 Getting Started (Hands‑on Code)

### 1. Installation & Setup

```bash
pip install llama-index -Uq
```

Set your OpenAI API key (used by default; local models are also supported – check future parts of the series).

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### 2. Prepare Your Data

Create a folder `data/` and place any files you want to ingest (PDF, txt, csv, …).  
For this example we use the **US Constitution** PDF.

```
project/
├── data/
│   └── constitution.pdf
└── your_script.py
```

### 3. The Famous Five‑Liner

This single block implements the whole RAG pipeline.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. Load documents from the data folder
documents = SimpleDirectoryReader("data").load_data()

# 2. Build the vector index (embeddings are created automatically)
index = VectorStoreIndex.from_documents(documents)

# 3. Create a query engine (retriever + response synthesizer)
query_engine = index.as_query_engine()

# 4. Ask a question
response = query_engine.query("What is the first article of the US Constitution about?")

print(response)
```

**Output:**  
`The first article establishes the legislative powers of Congress...`

### 4. Making the Index Persistent

If you have many documents, you don’t want to re‑ingest every time.  
Store the index to disk and reload it.

```python
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    # First run: load documents, create index, persist
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Later runs: load existing index from disk
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Query as before
query_engine = index.as_query_engine()
response = query_engine.query("What is the third article about?")
print(response)
```

### 5. Inspecting Documents & Nodes

The `SimpleDirectoryReader` returns a list of `Document` objects.  
Each document contains **metadata** and **text**.

```python
documents = SimpleDirectoryReader("data").load_data()
print(f"Number of documents: {len(documents)}")

# Look at the 4th document
doc = documents[4]
print(doc.metadata)  # file_name, page_label, etc.
print(doc.text[:500])  # first 500 characters
```

Example metadata:
```json
{
  "page_label": "4",
  "file_name": "constitution.pdf",
  "file_path": ".../data/constitution.pdf",
  "file_type": "application/pdf",
  "file_size": 12345,
  "creation_date": "2023-01-01",
  "last_modified_date": "2023-01-01"
}
```

### 6. Using LlamaParse for Complex Documents

`SimpleDirectoryReader` works well for text‑heavy PDFs, but for files with tables, images, or complex layouts, **LlamaParse** produces cleaner, structured text.

1. Sign up at [llamaindex.com](https://www.llamaindex.com) and get an API key.
2. Set it as an environment variable:

```python
os.environ["LLAMA_CLOUD_API_KEY"] = "your-llama-parse-api-key"
```

3. Use `LlamaParse` as the data connector:

```python
# Install the parser
# pip install llama-parse

import nest_asyncio
nest_asyncio.apply()  # needed for Jupyter / async env

from llama_parse import LlamaParse

# Replace SimpleDirectoryReader with LlamaParse
documents = LlamaParse(result_type="text").load_data("./data/constitution.pdf")

# Build index and query as before
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the first article about?")
print(response)
```

LlamaParse gives you **up to 1,000 pages/day for free** and returns beautifully structured text, preserving tables and layout.

---

## 📂 Supported File Types (SimpleDirectoryReader)

The `SimpleDirectoryReader` automatically recognises over **20 file formats**, including:

`.pdf`, `.docx`, `.txt`, `.csv`, `.tsv`, `.html`, `.md`, `.json`, `.xml`, `.ppt`, `.pptx`, `.jpg`, `.png`, `.mp3`, `.mp4` (via Unstructured), `.ipynb`, and more.

For a full, up‑to‑date list, see the [official docs](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/root.html).

---

## 🧠 Key Takeaways

- **LlamaIndex is a data‑centric framework** – it excels at ingesting, structuring, and indexing your private data for use with LLMs.
- The pipeline is built around **Documents → Nodes → Embeddings → Index**, with a strong emphasis on metadata and relationships.
- **Five lines of code** are enough to build a fully functional RAG system.
- For production, make the index **persistent** and consider **LlamaParse** for high‑quality extraction from complex documents.
- LlamaIndex provides advanced components (routers, retrievers, agents) that will be explored in future parts of the series.

---

## 🔗 Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [LlamaHub](https://llamahub.ai) – community data connectors & tools
- [LlamaCloud / LlamaParse](https://www.llamaindex.com) – cloud parsing & indexing
- [YouTube Video – Introduction to LlamaIndex](https://youtu.be/cCyYGYyCka4)

---

*This repository serves as a permanent deep‑dive artifact. It is based on the first part of an ongoing series – stay tuned for more advanced topics (custom retrievers, agents, local models, etc.).*
```

---

Simply create a new GitHub repository, copy the above into `README.md`, and you’ll have a complete, self‑contained LlamaIndex deep‑dive reference that you (or others) can use forever.
