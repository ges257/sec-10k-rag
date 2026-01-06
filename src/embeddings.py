# IBM 10-K RAG System - Embedding Model
# Generates embeddings for chunks and queries

from typing import List, Union, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingModel:
    """
    Embedding model wrapper for semantic search.

    Uses all-MiniLM-L6-v2 (384 dimensions) - same as Phase 3.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize (for cosine similarity)
            show_progress: Show progress bar
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings (N x dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            batch_size=batch_size
        )

        return embeddings

    def encode_chunks(
        self,
        chunks: List[dict],
        text_key: str = 'text'
    ) -> np.ndarray:
        """
        Encode chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries
            text_key: Key containing text in chunk dict

        Returns:
            Numpy array of embeddings
        """
        texts = [chunk[text_key] for chunk in chunks]
        return self.encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query text

        Returns:
            1D numpy array (dimension,)
        """
        embedding = self.encode([query], show_progress=False)
        return embedding[0]
