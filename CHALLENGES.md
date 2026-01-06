# Challenges: Building a Production RAG System

This document captures the key challenges overcome during development of the SEC 10-K RAG System.

---

## Challenge 1: Hybrid Search Hurt Performance

**Problem:** Adding BM25 to dense embeddings decreased accuracy from 76.7% to 66.7% (-10pp).

**Expected:** Hybrid search should capture both semantic similarity AND keyword matches.

**Actual:** For this domain, the MiniLM embeddings already captured semantics well. BM25's keyword matching added noise rather than signal.

**Diagnosis:**
- Evaluation questions were semantically clear
- Dense embeddings already captured domain concepts
- BM25 boosted irrelevant chunks with keyword overlap

**Solution:** Keep hybrid search but rely on cross-encoder to fix ranking errors.

**Lesson:** More components ≠ better results. Validate each addition.

---

## Challenge 2: Evaluation Design

**Problem:** Initial evaluation used LLM-based relevance judgments. Results were inconsistent and non-reproducible.

**Diagnosis:**
- LLMs gave different judgments on identical queries
- No ground truth to validate against
- Couldn't compare runs reliably

**Solution:** Switched to deterministic keyword/section matching:
- A chunk is relevant if it matches the target section OR contains 2+ gold keywords
- Hand-crafted 30 questions with explicit gold keywords and target sections

**Result:** Reproducible metrics. Can now measure true improvement.

---

## Challenge 3: Cross-Encoder Latency

**Problem:** Cross-encoder re-ranking added ~45ms latency (5ms → 49ms total).

**Diagnosis:**
- Cross-encoder scores each (query, document) pair independently
- Scoring 20 candidates = 20 forward passes
- CPU inference bottleneck

**Trade-off Decision:**
- 49ms is still acceptable for interactive use
- +23pp accuracy gain justifies the latency cost
- GPU would reduce to ~10ms if needed

**Mitigation:** Two-stage retrieval limits cross-encoder to top-20 candidates only.

---

## Challenge 4: Section Detection Accuracy

**Problem:** Automated section detection misclassified chunks.

**Diagnosis:**
- 10-K section headers vary by company
- "Item 1A" vs "Risk Factors" vs "Risks and Uncertainties"
- Naive keyword matching had high false positive rate

**Solution:** Multi-signal detection:
1. Section header patterns ("Item 1A", "Item 7")
2. Content keywords ("risk", "uncertainty", "may adversely")
3. Document position heuristics

**Result:** 96 chunks classified into 6 categories with acceptable accuracy.

---

## Challenge 5: Local LLM Quality

**Problem:** FLAN-T5-Small generates lower quality answers than GPT-4.

**Trade-off Decision:**
- Project goal: demonstrate retrieval quality, not generation quality
- Zero API cost enables unlimited experimentation
- Architecture supports drop-in replacement with larger models

**Mitigation:** Focus evaluation on retrieval metrics (Top-K, MRR), not answer quality.

---

## Key Insight: The Cross-Encoder MVP

The ablation results tell the full story:

| Configuration | Top-1 | Delta |
|--------------|-------|-------|
| Dense only (baseline) | 76.7% | — |
| + BM25 hybrid | 66.7% | -10pp |
| + Cross-encoder re-rank | 90.0% | +23.3pp |
| **Net improvement** | | **+13.3pp** |

**The cross-encoder's pairwise attention is doing all the heavy lifting.** It sees query and document together, understanding that "cybersecurity risk" isn't just about having both words present.

---

> See [LEARNINGS.md](LEARNINGS.md) for design decisions and trade-offs.
