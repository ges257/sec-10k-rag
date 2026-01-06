# IBM 10-K RAG System - Document Chunker
# Processes PDF into semantic chunks with metadata

import re
from typing import List, Dict, Optional
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from .section_detector import SectionDetector


class Chunker:
    """
    Document chunker with section metadata for 10-K filings.

    Features:
    - Word-based chunking with overlap
    - Section detection for each chunk
    - Metadata tracking (position, word count)
    """

    def __init__(
        self,
        chunk_size: int = 900,
        overlap: int = 150
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Words per chunk (default 900)
            overlap: Words overlap between chunks (default 150)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.section_detector = SectionDetector()

    def clean_text(self, text: str) -> str:
        """
        Clean text formatting (reused from Phase 2).

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Preserve paragraph breaks temporarily
        text = text.replace('\n\n', '[[PARAGRAPH_BREAK]]')

        # Normalize whitespace
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('[[PARAGRAPH_BREAK]]', '  ')

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove special unicode spaces
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)

        return text.strip()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted and cleaned text
        """
        if pdfplumber is None:
            raise ImportError("pdfplumber is required: pip install pdfplumber")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        all_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_text.append(page_text)

        # Join and clean
        full_text = "\n\n".join(all_text)
        return self.clean_text(full_text)

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: Full document text

        Returns:
            List of chunk dictionaries with metadata
        """
        words = text.split()
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(words):
            # Get chunk words
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Detect section
            section = self.section_detector.detect_section(chunk_text)

            # Create chunk with metadata
            chunk = {
                'id': chunk_id,
                'text': chunk_text,
                'section': section,
                'start_word': start,
                'end_word': end,
                'word_count': len(chunk_words),
            }

            chunks.append(chunk)
            chunk_id += 1

            # Move start with overlap
            start += self.chunk_size - self.overlap

            # Avoid infinite loop at end
            if end >= len(words):
                break

        return chunks

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Full pipeline: PDF -> cleaned text -> chunks with metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of chunk dictionaries
        """
        # Extract and clean text
        text = self.extract_text_from_pdf(pdf_path)

        # Create chunks
        chunks = self.chunk_text(text)

        return chunks

    def get_stats(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Statistics dictionary
        """
        sections = {}
        for chunk in chunks:
            section = chunk['section']
            sections[section] = sections.get(section, 0) + 1

        total_words = sum(c['word_count'] for c in chunks)

        return {
            'num_chunks': len(chunks),
            'total_words': total_words,
            'avg_words_per_chunk': total_words / len(chunks) if chunks else 0,
            'sections': sections,
        }
