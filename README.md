![Header](https://capsule-render.vercel.app/api?type=rect&color=0D1B2A&height=100&text=SEC%2010-K%20RAG%20System&fontSize=36&fontColor=A78BFA)

<div align="center">

**Hybrid Retrieval with Cross-Encoder Re-ranking for SEC Filing Analysis**

![Python](https://img.shields.io/badge/Python-3.11+-A3B8CC?style=flat-square&logo=python&logoColor=0D1B2A)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-A3B8CC?style=flat-square&logo=meta&logoColor=0D1B2A)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-A78BFA?style=flat-square&logo=huggingface&logoColor=0D1B2A)

[![Live Demo](https://img.shields.io/badge/Live_Demo-HuggingFace_Spaces-A78BFA?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/ges257/sec-10k-rag)

</div>

---

> Achieved Top-1 90% retrieval accuracy (+13pp over baseline) with sub-50ms latency. Hybrid Search (BM25 + Dense) with Cross-Encoder Re-ranker and local FLAN-T5 generation for zero marginal inference costs while maintaining full data sovereignty.

---

## Results

| Metric | Baseline | Full Pipeline | Delta |
|--------|----------|---------------|-------|
| **Top-1 Accuracy** | 76.7% | **90.0%** | +13.3pp |
| **Top-5 Accuracy** | 96.7% | **96.7%** | — |
| **Search Latency** | 5ms | **49ms** | — |
| **API Cost** | $0 | **$0** | — |

> **Key Finding:** Cross-encoder re-ranking provided the dominant accuracy lift (+23pp). Hybrid search alone actually hurt performance (-10pp), demonstrating that more components ≠ better results.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DOCUMENT PROCESSING                                             │
│  ─────────────────────                                          │
│  128-page PDF → 96 semantic chunks with section metadata        │
│  Sections: Revenue, Risk Factors, Strategy, Technology, etc.    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  HYBRID RETRIEVAL (Stage 1)                                     │
│  ─────────────────────────────                                  │
│  Dense: all-MiniLM-L6-v2 embeddings → FAISS                     │
│  Sparse: BM25Okapi for keyword matching                         │
│  Score: α × dense + (1-α) × sparse                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CROSS-ENCODER RE-RANKING (Stage 2)                             │
│  ─────────────────────────────────────                          │
│  Model: ms-marco-MiniLM-L-6-v2                                  │
│  Input: Top-20 candidates → Output: Top-5 re-ranked             │
│  Key: Pairwise attention sees query + doc together              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ANSWER GENERATION                                               │
│  ─────────────────────                                          │
│  Model: FLAN-T5-Small (local, CPU)                              │
│  Cost: $0 per query                                             │
│  Latency: ~200ms generation                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/ges257/sec-10k-rag.git
cd sec-10k-rag

# Install dependencies
pip install -r requirements.txt

# Add your PDF to data/
cp your_10k.pdf data/ibm_10k_2024.pdf

# Run Gradio demo
python app.py
# Open http://localhost:7860

# Or run FastAPI server
python api.py
# API docs at http://localhost:8000/docs
```

---

## Project Structure

```
sec-10k-rag/
├── src/
│   ├── chunker.py           # Document chunking with section metadata
│   ├── embeddings.py        # Sentence-transformer embeddings
│   ├── retriever.py         # Hybrid search + cross-encoder re-ranking
│   ├── generator.py         # FLAN-T5 answer generation
│   ├── index_manager.py     # Versioned FAISS index management
│   └── section_detector.py  # 10-K section classification
├── eval/
│   ├── eval_questions.json  # 30 ground-truth Q&A pairs
│   └── eval_harness.py      # Evaluation script
├── app.py                   # Gradio frontend
├── api.py                   # FastAPI backend
├── data/                    # Source PDFs (not in repo)
├── indices/                 # FAISS indices (not in repo)
├── README.md
├── ARCHITECTURE.md
├── CHALLENGES.md
└── LEARNINGS.md
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) |
| **Vector Store** | FAISS (IndexFlatIP) |
| **Sparse Search** | BM25Okapi |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Generator** | google/flan-t5-small |
| **Frontend** | Gradio |
| **Backend** | FastAPI |

---

## Documentation

- [System Architecture](ARCHITECTURE.md) — Detailed component design
- [Challenges & Problem-Solving](CHALLENGES.md) — What didn't work and why
- [Learnings & Trade-offs](LEARNINGS.md) — Design decisions and insights

---

## Author

**Gregory E. Schwartz**
- M.S. Artificial Intelligence (Yeshiva University)
- MBA (Cornell University)

---

![Footer](https://capsule-render.vercel.app/api?type=rect&color=0D1B2A&height=30&section=footer)
