# API Endpoint Flowcharts

Mermaid flowcharts for the two application endpoints. `GET /health` (liveness) is
omitted intentionally.

## `POST /upload` — ingest a PDF

```mermaid
flowchart TD
    A["Client<br/>POST /upload (multipart PDF)"] --> B{"filename ends<br/>with .pdf?"}
    B -- no --> B1["HTTP 400<br/>Only .pdf files supported"]
    B -- yes --> C["Save file to uploads/"]
    C --> D["engine.ingest(path)<br/>(run off event loop via threadpool)"]

    subgraph ingest ["RAGEngine.ingest"]
        D --> E["convert_pdf<br/>Docling parse → document"]
        E --> F["chunk_documents<br/>HybridChunker → page-tracked chunks"]
        F --> G{"any chunks?"}
        G -- no --> G1["chunks_added = 0"]
        G -- yes --> H["index_corpus<br/>embed + upsert to Pinecone<br/>(dense + sparse)"]
    end

    H --> I["HTTP 200<br/>{status, filename, chunks_added}"]
    G1 --> I
    D -. exception .-> Z["HTTP 500<br/>Ingestion failed"]
```

## `POST /query` — answer a question

```mermaid
flowchart TD
    A["Client<br/>POST /query {question, filenames?}"] --> B{"question<br/>non-empty?"}
    B -- no --> B1["HTTP 400<br/>question must not be empty"]
    B -- yes --> C["engine.answer(question, filenames)"]

    subgraph answer ["RAGEngine.answer"]
        C --> D["retrieve<br/>(threadpool)"]
        subgraph retrieval ["Retrieval chain"]
            D --> E["Dense top-30 + sparse top-30<br/>Pinecone, filtered to filenames"]
            E --> E2["Weighted RRF fusion"]
            E2 --> F["Rerank: Cohere rerank-v3.5<br/>→ bge-reranker-v2-m3 → fused order"]
            F --> G["Top-5 docs"]
        end
        G --> H{"any docs?"}
        H -- no --> H1["answer = 'no indexed<br/>documents' · empty lists"]
        H -- yes --> I["Format context<br/>(numbered [n] blocks)"]
        I --> J["Gemini → structured RAGAnswer<br/>(fallback models on 429/503)"]
        J --> K["Citations from doc metadata,<br/>flag the [n] the answer cites"]
    end

    K --> L["HTTP 200<br/>{answer, insights, next_questions,<br/>citations, model, reranker, timings}"]
    H1 --> L
    C -. exception .-> Z["HTTP 500<br/>Query failed"]
```

### Structured generation output (`RAGAnswer`)

```mermaid
flowchart LR
    R["RAGAnswer"] --> A["answer<br/>direct grounded answer"]
    R --> B["insights[]<br/>implications / next steps"]
    R --> C["next_questions[]<br/>suggested follow-ups"]
```
