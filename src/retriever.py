# IBM 10-K RAG System - Hybrid Retriever
# Combines BM25 + Dense search with Cross-Encoder re-ranking

from typing import List, Tuple, Optional, Dict
import numpy as np
import time

try:
    import faiss
except ImportError:
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from .embeddings import EmbeddingModel
from .section_detector import SectionDetector


class HybridRetriever:
    """
    Production-grade retriever with:
    - Dense search (FAISS)
    - Sparse search (BM25)
    - Hybrid combination (alpha blending)
    - Cross-encoder re-ranking
    - Section-aware boosting
    """

    def __init__(
        self,
        chunks: List[Dict],
        embedding_model: Optional[EmbeddingModel] = None,
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        use_cross_encoder: bool = True
    ):
        """
        Initialize hybrid retriever.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'section' keys
            embedding_model: Pre-initialized embedding model (creates new if None)
            cross_encoder_model: Cross-encoder model name for re-ranking
            use_cross_encoder: Whether to use cross-encoder re-ranking
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        self.chunks = chunks
        self.section_detector = SectionDetector()

        # Initialize embedding model
        if embedding_model is None:
            embedding_model = EmbeddingModel()
        self.embedding_model = embedding_model

        # Build dense index (FAISS)
        print("Building dense index (FAISS)...")
        self._build_dense_index()

        # Build sparse index (BM25)
        print("Building sparse index (BM25)...")
        self._build_sparse_index()

        # Load cross-encoder for re-ranking
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder = None
        if use_cross_encoder:
            print("Loading cross-encoder for re-ranking...")
            self._load_cross_encoder(cross_encoder_model)

        print(f"Retriever ready: {len(chunks)} chunks indexed")

    def _build_dense_index(self):
        """Build FAISS index for dense retrieval."""
        # Generate embeddings
        self.embeddings = self.embedding_model.encode_chunks(self.chunks)

        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim

        # Add normalized vectors
        embeddings_normalized = self.embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        self.faiss_index.add(embeddings_normalized)

    def _build_sparse_index(self):
        """Build BM25 index for sparse retrieval."""
        if BM25Okapi is None:
            print("Warning: rank_bm25 not installed, sparse search disabled")
            self.bm25 = None
            return

        # Tokenize chunks
        tokenized = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _load_cross_encoder(self, model_name: str):
        """Load cross-encoder for re-ranking."""
        if CrossEncoder is None:
            print("Warning: CrossEncoder not available, re-ranking disabled")
            self.use_cross_encoder = False
            return

        try:
            self.cross_encoder = CrossEncoder(model_name)
        except Exception as e:
            print(f"Warning: Failed to load cross-encoder: {e}")
            self.use_cross_encoder = False

    def dense_search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[Dict, float]]:
        """
        Dense-only search using FAISS.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        # Encode query
        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # Valid index
                results.append((self.chunks[idx], float(score)))

        return results

    def sparse_search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[Dict, float]]:
        """
        Sparse-only search using BM25.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25 is None:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx])))

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        alpha: float = 0.5
    ) -> List[Tuple[Dict, float]]:
        """
        Hybrid search combining dense and sparse.

        Args:
            query: Query text
            top_k: Number of results
            alpha: Weight for dense scores (1-alpha for sparse)

        Returns:
            List of (chunk, score) tuples
        """
        # Get dense scores
        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        dense_scores, _ = self.faiss_index.search(query_embedding, len(self.chunks))
        dense_scores = (dense_scores[0] + 1) / 2  # Normalize to [0, 1]

        # Get sparse scores
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
            # Normalize sparse scores
            max_sparse = max(sparse_scores) if max(sparse_scores) > 0 else 1
            sparse_scores = sparse_scores / max_sparse
        else:
            sparse_scores = np.zeros(len(self.chunks))

        # Combine scores
        hybrid_scores = []
        for i in range(len(self.chunks)):
            combined = alpha * dense_scores[i] + (1 - alpha) * sparse_scores[i]
            hybrid_scores.append((i, combined))

        # Sort and return top-k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in hybrid_scores[:top_k]:
            results.append((self.chunks[idx], float(score)))

        return results

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict, float]],
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Re-rank candidates using cross-encoder.

        Args:
            query: Query text
            candidates: Initial retrieval results
            top_k: Number of final results

        Returns:
            Re-ranked list of (chunk, score) tuples
        """
        if not self.use_cross_encoder or self.cross_encoder is None:
            return candidates[:top_k]

        # Create pairs for cross-encoder
        pairs = [(query, chunk['text']) for chunk, _ in candidates]

        # Score with cross-encoder
        cross_scores = self.cross_encoder.predict(pairs)

        # Combine with original scores (optional - can use cross-scores only)
        reranked = list(zip(candidates, cross_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with cross-encoder scores
        results = []
        for (chunk, _), ce_score in reranked[:top_k]:
            results.append((chunk, float(ce_score)))

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        top_k_initial: int = 20,
        alpha: float = 0.5,
        use_rerank: bool = True,
        apply_section_boost: bool = True,
        section_boost_factor: float = 1.3
    ) -> Tuple[List[Tuple[Dict, float]], Dict]:
        """
        Full search pipeline: Hybrid -> Re-rank -> Section boost.

        Args:
            query: Query text
            top_k: Final number of results
            top_k_initial: Initial candidates before re-ranking
            alpha: Hybrid search weight (dense vs sparse)
            use_rerank: Whether to use cross-encoder re-ranking
            apply_section_boost: Whether to apply section boosts
            section_boost_factor: Boost multiplier for matching sections

        Returns:
            Tuple of (results, metrics)
            - results: List of (chunk, score) tuples
            - metrics: Dict with timing and method info
        """
        start_time = time.time()
        metrics = {
            'query': query,
            'method': 'hybrid',
            'top_k': top_k,
            'alpha': alpha,
        }

        # Stage 1: Hybrid search
        stage1_start = time.time()
        candidates = self.hybrid_search(query, top_k=top_k_initial, alpha=alpha)
        metrics['stage1_time_ms'] = (time.time() - stage1_start) * 1000
        metrics['stage1_candidates'] = len(candidates)

        # Stage 2: Cross-encoder re-ranking
        if use_rerank and self.use_cross_encoder:
            stage2_start = time.time()
            results = self.rerank(query, candidates, top_k=top_k)
            metrics['stage2_time_ms'] = (time.time() - stage2_start) * 1000
            metrics['reranked'] = True
        else:
            results = candidates[:top_k]
            metrics['reranked'] = False

        # Stage 3: Section boost
        if apply_section_boost:
            results = self.section_detector.apply_section_boost(
                query, results, boost_factor=section_boost_factor
            )
            metrics['section_boosted'] = True
        else:
            metrics['section_boosted'] = False

        metrics['total_time_ms'] = (time.time() - start_time) * 1000
        metrics['num_results'] = len(results)

        return results, metrics

    def baseline_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Baseline dense-only search (for comparison).

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        return self.dense_search(query, top_k=top_k)
