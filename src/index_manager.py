# IBM 10-K RAG System - Versioned Index Manager
# Manages timestamped FAISS indices for production deployments

import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None


class VersionedIndexManager:
    """
    Manages versioned FAISS indices for production deployments.

    Features:
    - Timestamped index versions
    - "latest" symlink for current version
    - Metadata tracking (chunks, config, stats)
    - Easy rollback to previous versions
    """

    def __init__(self, base_dir: str = "indices"):
        """
        Initialize index manager.

        Args:
            base_dir: Directory to store versioned indices
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        faiss_index: 'faiss.Index',
        chunks: List[Dict],
        embeddings: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save index with version timestamp.

        Args:
            faiss_index: FAISS index to save
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
            metadata: Optional additional metadata

        Returns:
            Version string (timestamp)
        """
        # Generate version timestamp
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = version_dir / "index.faiss"
        faiss.write_index(faiss_index, str(index_path))

        # Save chunks
        chunks_path = version_dir / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        # Save embeddings
        embeddings_path = version_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        # Save metadata
        meta = {
            "version": version,
            "created": datetime.now().isoformat(),
            "num_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "index_type": type(faiss_index).__name__,
            **(metadata or {})
        }

        # Add section distribution
        sections = {}
        for chunk in chunks:
            section = chunk.get('section', 'other')
            sections[section] = sections.get(section, 0) + 1
        meta["sections"] = sections

        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Update latest symlink
        self._update_latest_symlink(version)

        print(f"Saved index version: {version}")
        print(f"  - Chunks: {len(chunks)}")
        print(f"  - Embeddings: {embeddings.shape}")
        print(f"  - Location: {version_dir}")

        return version

    def _update_latest_symlink(self, version: str):
        """Update 'latest' symlink to point to new version."""
        latest_link = self.base_dir / "latest"

        # Remove existing symlink
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink (relative)
        latest_link.symlink_to(version)

    def load(
        self,
        version: str = "latest"
    ) -> Tuple['faiss.Index', List[Dict], np.ndarray, Dict]:
        """
        Load specific version or latest.

        Args:
            version: Version string or "latest"

        Returns:
            Tuple of (faiss_index, chunks, embeddings, metadata)
        """
        version_dir = self.base_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(f"Index version not found: {version}")

        # If it's a symlink, resolve it
        if version_dir.is_symlink():
            version_dir = version_dir.resolve()

        # Load FAISS index
        index_path = version_dir / "index.faiss"
        faiss_index = faiss.read_index(str(index_path))

        # Load chunks
        chunks_path = version_dir / "chunks.pkl"
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        # Load embeddings
        embeddings_path = version_dir / "embeddings.npy"
        embeddings = np.load(embeddings_path)

        # Load metadata
        meta_path = version_dir / "metadata.json"
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        print(f"Loaded index version: {metadata.get('version', version)}")
        print(f"  - Chunks: {len(chunks)}")
        print(f"  - Created: {metadata.get('created', 'unknown')}")

        return faiss_index, chunks, embeddings, metadata

    def list_versions(self) -> List[Dict]:
        """
        List all available versions with metadata.

        Returns:
            List of version info dictionaries
        """
        versions = []

        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name != "latest":
                meta_path = item / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    versions.append(meta)

        # Sort by version (newest first)
        versions.sort(key=lambda x: x.get('version', ''), reverse=True)

        return versions

    def get_latest_version(self) -> Optional[str]:
        """Get the latest version string."""
        latest_link = self.base_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                return latest_link.resolve().name
            return "latest"
        return None

    def delete_version(self, version: str) -> bool:
        """
        Delete a specific version.

        Args:
            version: Version string to delete

        Returns:
            True if deleted, False if not found
        """
        if version == "latest":
            print("Cannot delete 'latest' - delete specific version instead")
            return False

        version_dir = self.base_dir / version

        if not version_dir.exists():
            return False

        import shutil
        shutil.rmtree(version_dir)

        # Update latest if needed
        latest_link = self.base_dir / "latest"
        if latest_link.is_symlink():
            current_latest = latest_link.resolve().name
            if current_latest == version:
                # Point to most recent remaining version
                versions = self.list_versions()
                if versions:
                    self._update_latest_symlink(versions[0]['version'])
                else:
                    latest_link.unlink()

        print(f"Deleted index version: {version}")
        return True

    def cleanup_old_versions(self, keep_count: int = 5) -> int:
        """
        Delete old versions, keeping only the most recent.

        Args:
            keep_count: Number of versions to keep

        Returns:
            Number of versions deleted
        """
        versions = self.list_versions()

        if len(versions) <= keep_count:
            return 0

        # Delete oldest versions
        deleted = 0
        for version_info in versions[keep_count:]:
            version = version_info.get('version')
            if version and self.delete_version(version):
                deleted += 1

        return deleted
