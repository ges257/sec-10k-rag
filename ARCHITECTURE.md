# IBM 10-K RAG System - Architecture

## System Overview

```mermaid
flowchart LR
    subgraph UI["User Interface"]
        A1["Gradio Frontend"]
        A2["app.py"]
        A1 --> A2
    end

    subgraph API["Backend"]
        B1["FastAPI"]
        B2["api.py"]
        B1 --> B2
    end

    subgraph Retrieval["Retriever Pipeline"]
        C1["FAISS (Dense)"]
        C2["BM25 (Sparse)"]
        C3["Cross-Encoder"]
    end

    subgraph Gen["Generator"]
        D1["FLAN-T5 LLM"]
    end

    UI --> API --> Retrieval --> Gen

    style UI fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style API fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Retrieval fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Gen fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    linkStyle 0,1,2,3,4 stroke:#A78BFA,stroke-width:2px
```

## Data Flow

### 1. Document Processing Pipeline

```mermaid
flowchart TB
    subgraph Input["Input"]
        A1["PDF Document"]
    end

    subgraph Parse["Parsing"]
        B1["PDFPlumber"]
        B2["Extract text"]
        B1 --> B2
    end

    subgraph Classify["Classification"]
        C1["Section Detection"]
        C2["Risk Factors, Revenue, Strategy..."]
        C1 --> C2
    end

    subgraph Chunk["Chunking"]
        D1["900 words, 150 overlap"]
        D2["Preserve section metadata"]
        D1 --> D2
    end

    subgraph Embed["Embedding"]
        E1["all-MiniLM-L6-v2"]
        E2["384 dimensions"]
        E1 --> E2
    end

    subgraph Store["Storage"]
        F1["FAISS Index"]
        F2["IndexFlatIP (cosine)"]
        F1 --> F2
    end

    Input --> Parse --> Classify --> Chunk --> Embed --> Store

    style Input fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Parse fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Classify fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Chunk fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Embed fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Store fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    linkStyle 0,1,2,3,4,5,6,7,8,9 stroke:#A78BFA,stroke-width:2px
```

### 2. Query Processing Pipeline

```mermaid
flowchart TB
    subgraph Query["User Query"]
        Q1["Input question"]
    end

    subgraph Search["Parallel Search"]
        S1["Dense Search (FAISS)"]
        S2["Sparse Search (BM25)"]
    end

    subgraph Fusion["Hybrid Fusion"]
        F1["α × dense + (1-α) × sparse"]
        F2["Top-20 candidates"]
        F1 --> F2
    end

    subgraph Rerank["Cross-Encoder Re-ranking"]
        R1["ms-marco-MiniLM-L-6-v2"]
        R2["Top-5 re-ranked"]
        R1 --> R2
    end

    subgraph Boost["Section Boost"]
        B1["Query-relevant sections"]
        B2["+30% score boost"]
        B1 --> B2
    end

    subgraph Generate["Answer Generation"]
        G1["FLAN-T5 Generator"]
        G2["Final Answer"]
        G1 --> G2
    end

    Query --> S1
    Query --> S2
    S1 --> Fusion
    S2 --> Fusion
    Fusion --> Rerank --> Boost --> Generate

    style Query fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Search fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Fusion fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Rerank fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Boost fill:#1a1a2e,stroke:#A78BFA,color:#A3B8CC
    style Generate fill:#A78BFA,stroke:#A78BFA,color:#0D1B2A
    linkStyle 0,1,2,3,4,5,6,7,8,9,10 stroke:#A78BFA,stroke-width:2px
```

## Component Details

### 1. Document Chunker (`src/chunker.py`)

**Purpose**: Convert PDF into searchable chunks with metadata.

**Key Design Decisions**:
- Word-based chunking (900 words) instead of character-based for semantic coherence
- 150-word overlap to prevent information loss at boundaries
- Section metadata attached to each chunk for downstream boosting

**Input**: PDF file path
**Output**: List of chunk dictionaries with `text`, `section`, `chunk_id`

### 2. Section Detector (`src/section_detector.py`)

**Purpose**: Classify 10-K sections and enable query-aware boosting.

**Sections Detected**:
- `risk_factors` - Item 1A: Risk Factors
- `revenue` - Revenue, Financial Data
- `strategy` - Business Strategy, Growth Plans
- `technology` - R&D, AI, Cloud, Quantum
- `executive` - CEO Letter, Management Discussion
- `business` - Business Overview, Operations

**Query Hints**: Maps keywords to expected sections:
- "risk", "cyber", "regulatory" → risk_factors
- "revenue", "profit", "margin" → revenue
- "strategy", "growth", "cloud" → strategy

### 3. Embedding Model (`src/embeddings.py`)

**Purpose**: Generate dense vector representations.

**Model**: `all-MiniLM-L6-v2`
- 384 dimensions
- Fast inference on CPU
- Good balance of speed and quality

**Optimizations**:
- Batch encoding for chunks
- L2 normalization for cosine similarity

### 4. Hybrid Retriever (`src/retriever.py`)

**Purpose**: Two-stage retrieval with hybrid search and re-ranking.

**Stage 1: Hybrid Search**
```python
# Dense: FAISS cosine similarity
# Sparse: BM25 keyword matching
hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
```

Default alpha = 0.5 (equal weight)

**Stage 2: Cross-Encoder Re-ranking**
```python
# Re-rank top-20 candidates to top-5
cross_encoder.predict([(query, chunk_text) for chunk in candidates])
```

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Stage 3: Section Boosting**
```python
# Boost chunks from query-relevant sections
if chunk.section in query_hints:
    score *= 1.3
```

### 5. Answer Generator (`src/generator.py`)

**Purpose**: Generate natural language answers from retrieved context.

**Model**: `google/flan-t5-small`
- Zero API cost
- CPU-friendly
- Instruction-tuned for Q&A

**Prompt Template**:
```
Based only on the following context from IBM's 10-K SEC filing,
answer the question.

Context:
[Source 1 - risk_factors]: ...
[Source 2 - revenue]: ...

Question: What are IBM's main revenue segments?

Answer:
```

### 6. Index Manager (`src/index_manager.py`)

**Purpose**: Production-ready versioned index storage.

**Features**:
- Timestamped versions (YYYYMMDD_HHMMSS)
- "latest" symlink for current version
- Metadata tracking (chunk count, sections, etc.)
- Rollback capability

**Directory Structure**:
```
indices/
├── 20251205_143022/
│   ├── index.faiss
│   ├── chunks.pkl
│   ├── embeddings.npy
│   └── metadata.json
├── 20251205_160045/
│   └── ...
└── latest -> 20251205_160045
```

## Evaluation Framework

### Metrics

1. **Top-K Accuracy**: % of queries where relevant chunk appears in top K results
2. **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first relevant result
3. **Latency**: End-to-end search time in milliseconds

### Ground Truth

30 hand-crafted questions covering:
- Risk Factors (8 questions)
- Revenue (6 questions)
- Strategy (4 questions)
- Technology (5 questions)
- Executive (2 questions)
- Business (3 questions)
- Financial (2 questions)

### Relevance Criteria

A chunk is considered relevant if:
1. Section matches target section, OR
2. 2+ gold keywords appear in chunk text

## Performance Analysis

### Why Hybrid Search Helps

| Query Type | Dense Only | Hybrid |
|------------|------------|--------|
| Semantic ("business strategy") | Good | Good |
| Keyword ("watsonx AI") | Poor | Good |
| Entity ("Red Hat acquisition") | Poor | Good |

BM25 captures exact keyword matches that dense embeddings may miss.

### Why Re-ranking Helps

Cross-encoder sees query AND document together (cross-attention), enabling:
- Better understanding of query intent
- Finer-grained relevance distinctions
- Higher precision in top-5 results

### Latency Breakdown

| Stage | Time |
|-------|------|
| Dense Search (FAISS) | ~10ms |
| Sparse Search (BM25) | ~20ms |
| Cross-Encoder (20 pairs) | ~100ms |
| Section Boost | ~1ms |
| LLM Generation | ~200ms |
| **Total** | **~330ms** |

## Future Improvements

1. **GPU Acceleration**: Move embeddings and cross-encoder to GPU
2. **Quantized Indices**: IVF-PQ for larger document sets
3. **Better LLM**: Upgrade to Llama-2 or Mistral for better answers
4. **Multi-document**: Support multiple 10-K filings
5. **Streaming**: Stream LLM responses for better UX
