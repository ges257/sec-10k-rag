# Learnings: RAG System Design

Key insights from building a production-grade retrieval system for SEC filing analysis.

---

## 1. More Components ≠ Better Results

**Assumption:** Hybrid search (BM25 + dense) should always beat single-method retrieval.

**Reality:** Dense-only achieved 76.7%, hybrid dropped to 66.7%.

**Why:** The MiniLM embeddings already captured domain semantics well. BM25's exact keyword matching added noise for queries that were semantically clear.

**Lesson:** Validate each component independently. Ablation studies are essential.

---

## 2. Cross-Encoder Attention is the Differentiator

**Bi-encoders:** Embed query and document independently, compare with dot product.

**Cross-encoders:** See query AND document together with cross-attention.

**Why it matters:** Understanding "cybersecurity risk" requires seeing how "cybersecurity" relates to "risk" in context. Bi-encoders can't do this.

**Impact:** +23pp accuracy from cross-encoder alone.

---

## 3. Two-Stage Retrieval for Speed + Precision

**Problem:** Cross-encoders are slow (can't score entire corpus).

**Solution:** Two-stage pipeline:
1. **Fast retrieval:** FAISS returns top-20 in 5ms
2. **Precise re-ranking:** Cross-encoder scores 20 candidates in 44ms

**Result:** Best of both worlds — sub-50ms total latency with high precision.

---

## 4. Evaluation Design Determines Success

**Bad approach:** LLM-based relevance judgments
- Inconsistent across runs
- Non-reproducible
- Can't compare methods fairly

**Good approach:** Deterministic keyword/section matching
- Reproducible metrics
- Clear ground truth
- Enables valid ablation studies

**Lesson:** Invest in evaluation infrastructure before optimizing retrieval.

---

## 5. Local LLM Trade-offs

**Why FLAN-T5-Small:**
- Zero API cost
- Full data sovereignty (critical for SEC filings)
- Fast iteration during development

**What you lose:**
- Answer quality (vs. GPT-4)
- Reasoning depth
- Multi-step synthesis

**When to upgrade:** Production deployment where answer quality matters. Architecture supports drop-in replacement.

---

## 6. Section Metadata is Underrated

**Without section awareness:** All chunks weighted equally.

**With section awareness:**
- "risk" queries boost Risk Factors section
- "revenue" queries boost Financial sections
- Reduces false positives significantly

**Implementation:** Query intent detection + section boost multiplier (1.3x).

---

## 7. Index Versioning for Production

**Problem:** Re-indexing a document shouldn't break production.

**Solution:** Versioned indices with timestamps:
- Save new version alongside current
- Atomic switchover via symlink
- Rollback by pointing to previous version

**Lesson:** Production thinking even in portfolio projects signals seniority.

---

## Design Decision Summary

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Search method | Hybrid + re-rank | Dense only | Cross-encoder fixes hybrid's mistakes |
| Re-ranker | ms-marco-MiniLM | None | +23pp accuracy worth 44ms latency |
| Generator | FLAN-T5-Small | GPT-4 | Zero cost, data sovereignty |
| Eval method | Keyword/section | LLM judging | Reproducibility |
| Vector store | FAISS local | Pinecone | Single doc, no API cost |

---

## What I'd Do Differently

1. **Start with eval harness** — Wasted time optimizing without valid metrics
2. **Test components in isolation** — Hybrid search failure was surprising
3. **GPU from the start** — Cross-encoder latency acceptable but could be better

---

## Scaling Considerations

| Scenario | Current | Upgrade Path |
|----------|---------|--------------|
| Multiple documents | Single index | Merge with doc-id metadata |
| Higher QPS | Serial | Batch queries, GPU inference |
| Lower latency | 49ms | Remove re-ranking for non-critical |
| Better answers | FLAN-T5-Small | Mistral-7B, Llama 3 |

---

> See [CHALLENGES.md](CHALLENGES.md) for problem-solving details.
