# IBM 10-K RAG System - Evaluation Harness
# Measures retrieval quality with Top-K accuracy metrics

import json
import time
from typing import List, Dict, Callable, Tuple, Optional
from pathlib import Path
from datetime import datetime


class EvalHarness:
    """
    Evaluation harness for measuring retrieval quality.

    Metrics:
    - Top-1 Accuracy: % of queries where best result is relevant
    - Top-3 Accuracy: % of queries where top 3 includes relevant result
    - Top-5 Accuracy: % of queries where top 5 includes relevant result
    - Mean Reciprocal Rank (MRR)
    - Latency statistics
    """

    def __init__(self, eval_questions_path: str = "eval/eval_questions.json"):
        """
        Initialize evaluation harness.

        Args:
            eval_questions_path: Path to evaluation questions JSON
        """
        self.eval_questions_path = Path(eval_questions_path)
        self.questions = self._load_questions()

    def _load_questions(self) -> List[Dict]:
        """Load evaluation questions from JSON."""
        if not self.eval_questions_path.exists():
            raise FileNotFoundError(f"Eval questions not found: {self.eval_questions_path}")

        with open(self.eval_questions_path, 'r') as f:
            return json.load(f)

    def _is_relevant(
        self,
        chunk: Dict,
        question: Dict,
        check_section: bool = True,
        check_keywords: bool = True
    ) -> bool:
        """
        Check if a chunk is relevant to a question.

        Args:
            chunk: Retrieved chunk dictionary
            question: Evaluation question dictionary
            check_section: Whether to check section match
            check_keywords: Whether to check keyword presence

        Returns:
            True if chunk is relevant
        """
        chunk_text = chunk.get('text', '').lower()
        chunk_section = chunk.get('section', '')

        # Check section match
        if check_section:
            target_section = question.get('target_section', '')
            if target_section and chunk_section == target_section:
                return True

        # Check keyword presence
        if check_keywords:
            keywords = question.get('gold_keywords', [])
            matches = sum(1 for kw in keywords if kw.lower() in chunk_text)
            # Consider relevant if >= 2 keywords match
            if matches >= 2:
                return True

        return False

    def evaluate_retriever(
        self,
        retriever_fn: Callable,
        top_k_values: List[int] = [1, 3, 5],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate a retriever function on the eval set.

        Args:
            retriever_fn: Function(query, top_k) -> List[(chunk, score)]
            top_k_values: List of K values to evaluate
            verbose: Print progress

        Returns:
            Dictionary of evaluation metrics
        """
        max_k = max(top_k_values)
        results = {f'top_{k}': 0 for k in top_k_values}
        reciprocal_ranks = []
        latencies = []

        for i, question in enumerate(self.questions):
            query = question['question']

            # Time the retrieval
            start = time.time()
            retrieved = retriever_fn(query, max_k)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            # Find first relevant result
            first_relevant_rank = None
            for rank, (chunk, score) in enumerate(retrieved, 1):
                if self._is_relevant(chunk, question):
                    first_relevant_rank = rank
                    break

            # Update top-k metrics
            for k in top_k_values:
                if first_relevant_rank is not None and first_relevant_rank <= k:
                    results[f'top_{k}'] += 1

            # Update MRR
            if first_relevant_rank is not None:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(self.questions)} questions...")

        # Calculate final metrics
        n = len(self.questions)
        metrics = {
            'num_questions': n,
            'top_k_accuracy': {
                f'top_{k}': round(results[f'top_{k}'] / n * 100, 1)
                for k in top_k_values
            },
            'mrr': round(sum(reciprocal_ranks) / n, 4),
            'latency_ms': {
                'mean': round(sum(latencies) / n, 1),
                'min': round(min(latencies), 1),
                'max': round(max(latencies), 1),
            }
        }

        return metrics

    def compare_retrievers(
        self,
        retrievers: Dict[str, Callable],
        top_k_values: List[int] = [1, 3, 5]
    ) -> Dict:
        """
        Compare multiple retriever configurations.

        Args:
            retrievers: Dict of {name: retriever_fn}
            top_k_values: K values to evaluate

        Returns:
            Dictionary with comparison results
        """
        print(f"Comparing {len(retrievers)} retriever configurations...")
        print(f"Evaluation set: {len(self.questions)} questions")
        print()

        comparison = {
            'timestamp': datetime.now().isoformat(),
            'num_questions': len(self.questions),
            'results': {}
        }

        for name, retriever_fn in retrievers.items():
            print(f"Evaluating: {name}")
            metrics = self.evaluate_retriever(retriever_fn, top_k_values)
            comparison['results'][name] = metrics
            print(f"  Top-1: {metrics['top_k_accuracy']['top_1']}%")
            print(f"  Top-5: {metrics['top_k_accuracy']['top_5']}%")
            print(f"  MRR: {metrics['mrr']}")
            print(f"  Latency: {metrics['latency_ms']['mean']}ms")
            print()

        # Calculate improvements
        if len(retrievers) >= 2:
            names = list(retrievers.keys())
            baseline_name = names[0]
            comparison['improvements'] = {}

            for name in names[1:]:
                baseline_top1 = comparison['results'][baseline_name]['top_k_accuracy']['top_1']
                current_top1 = comparison['results'][name]['top_k_accuracy']['top_1']
                improvement = current_top1 - baseline_top1

                comparison['improvements'][f'{name}_vs_{baseline_name}'] = {
                    'top_1_delta': round(improvement, 1),
                    'relative_improvement': round(improvement / baseline_top1 * 100, 1) if baseline_top1 > 0 else 0
                }

        return comparison

    def evaluate_by_section(
        self,
        retriever_fn: Callable,
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate retriever broken down by target section.

        Args:
            retriever_fn: Retriever function
            top_k: K value for top-k accuracy

        Returns:
            Dictionary with per-section metrics
        """
        section_results = {}

        for question in self.questions:
            section = question.get('target_section', 'other')
            if section not in section_results:
                section_results[section] = {'total': 0, 'correct': 0}

            query = question['question']
            retrieved = retriever_fn(query, top_k)

            # Check if any top-k result is relevant
            is_correct = any(
                self._is_relevant(chunk, question)
                for chunk, _ in retrieved
            )

            section_results[section]['total'] += 1
            if is_correct:
                section_results[section]['correct'] += 1

        # Calculate accuracy per section
        metrics = {}
        for section, counts in section_results.items():
            if counts['total'] > 0:
                accuracy = round(counts['correct'] / counts['total'] * 100, 1)
                metrics[section] = {
                    'total': counts['total'],
                    'correct': counts['correct'],
                    'accuracy': accuracy
                }

        return metrics

    def save_results(self, results: Dict, output_path: str = "eval/eval_results.json"):
        """Save evaluation results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


def run_evaluation(
    retriever,
    output_path: str = "eval/eval_results.json"
):
    """
    Convenience function to run full evaluation.

    Args:
        retriever: HybridRetriever instance
        output_path: Path to save results
    """
    harness = EvalHarness()

    # Define retriever configurations
    def baseline_fn(query, top_k):
        return retriever.baseline_search(query, top_k)

    def hybrid_fn(query, top_k):
        return retriever.hybrid_search(query, top_k)

    def hybrid_rerank_fn(query, top_k):
        results, _ = retriever.search(
            query,
            top_k=top_k,
            use_rerank=True,
            apply_section_boost=False
        )
        return results

    def full_fn(query, top_k):
        results, _ = retriever.search(
            query,
            top_k=top_k,
            use_rerank=True,
            apply_section_boost=True
        )
        return results

    retrievers = {
        'baseline_dense': baseline_fn,
        'hybrid_bm25_dense': hybrid_fn,
        'hybrid_with_rerank': hybrid_rerank_fn,
        'full_pipeline': full_fn,
    }

    # Run comparison
    results = harness.compare_retrievers(retrievers)

    # Add per-section breakdown for full pipeline
    results['section_breakdown'] = harness.evaluate_by_section(full_fn)

    # Save results
    harness.save_results(results, output_path)

    return results


if __name__ == "__main__":
    # Example usage - requires initialized retriever
    print("Evaluation harness ready.")
    print("Usage: from eval.eval_harness import run_evaluation")
    print("       results = run_evaluation(retriever)")
