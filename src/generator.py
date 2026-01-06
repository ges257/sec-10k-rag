# IBM 10-K RAG System - Answer Generator
# LLM-based answer generation from retrieved context

from typing import List, Dict, Tuple, Optional

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    pipeline = None
    torch = None


class AnswerGenerator:
    """
    LLM-based answer generator using local FLAN-T5.

    Features:
    - Zero API cost (local model)
    - CPU-optimized (flan-t5-small)
    - Context-grounded answers
    """

    def __init__(
        self,
        model_name: str = 'google/flan-t5-small',
        device: Optional[str] = None,
        max_context_length: int = 2000,
        max_answer_length: int = 150
    ):
        """
        Initialize answer generator.

        Args:
            model_name: HuggingFace model name (default: flan-t5-small for CPU)
            device: Device ('cuda', 'cpu', or None for auto)
            max_context_length: Max characters of context to include
            max_answer_length: Max tokens in generated answer
        """
        if AutoTokenizer is None:
            raise ImportError(
                "transformers is required: pip install transformers"
            )

        self.model_name = model_name
        self.max_context_length = max_context_length
        self.max_answer_length = max_answer_length

        # Determine device
        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device

        print(f"Loading LLM: {model_name} on {device}...")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Create pipeline for easier generation
        self.pipe = pipeline(
            'text2text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == 'cuda' else -1,
            max_length=max_answer_length
        )

        print("LLM ready")

    def build_prompt(
        self,
        question: str,
        results: List[Tuple[Dict, float]],
        max_sources: int = 5
    ) -> str:
        """
        Build prompt with question and retrieved context.

        Args:
            question: User's question
            results: Retrieved (chunk, score) tuples
            max_sources: Maximum sources to include

        Returns:
            Formatted prompt string
        """
        # Build context from top results
        context_parts = []
        chars_used = 0

        for i, (chunk, score) in enumerate(results[:max_sources]):
            chunk_text = chunk['text']

            # Truncate if needed
            available = self.max_context_length - chars_used
            if len(chunk_text) > available:
                chunk_text = chunk_text[:available] + "..."

            section = chunk.get('section', 'unknown')
            context_parts.append(f"[Source {i+1} - {section}]: {chunk_text}")
            chars_used += len(chunk_text)

            if chars_used >= self.max_context_length:
                break

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = f"""Based only on the following context from IBM's 10-K SEC filing, answer the question.
If the answer cannot be found in the context, say "I cannot find this information in the provided context."

Context:
{context}

Question: {question}

Answer:"""

        return prompt

    def generate(
        self,
        question: str,
        results: List[Tuple[Dict, float]],
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> str:
        """
        Generate answer from question and retrieved context.

        Args:
            question: User's question
            results: Retrieved (chunk, score) tuples
            temperature: Sampling temperature
            num_beams: Beam search width

        Returns:
            Generated answer string
        """
        # Build prompt
        prompt = self.build_prompt(question, results)

        # Generate
        with torch.no_grad():
            outputs = self.pipe(
                prompt,
                max_length=self.max_answer_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        answer = outputs[0]['generated_text']
        return answer.strip()

    def generate_with_sources(
        self,
        question: str,
        results: List[Tuple[Dict, float]],
        include_sources: bool = True
    ) -> Dict:
        """
        Generate answer with source citations.

        Args:
            question: User's question
            results: Retrieved (chunk, score) tuples
            include_sources: Whether to include source info

        Returns:
            Dict with 'answer', 'sources', 'prompt'
        """
        answer = self.generate(question, results)

        output = {
            'question': question,
            'answer': answer,
        }

        if include_sources:
            sources = []
            for i, (chunk, score) in enumerate(results):
                sources.append({
                    'rank': i + 1,
                    'chunk_id': chunk.get('id', i),
                    'section': chunk.get('section', 'unknown'),
                    'score': round(score, 4),
                    'preview': chunk['text'][:200] + '...',
                })
            output['sources'] = sources

        return output
