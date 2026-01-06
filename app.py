# IBM 10-K RAG System - Gradio Frontend
# Interactive Q&A interface with retrieval options

import gradio as gr
import json
import time
from pathlib import Path

# Import system components
from src.chunker import DocumentChunker
from src.retriever import HybridRetriever
from src.generator import AnswerGenerator
from src.index_manager import VersionedIndexManager


# Global state
retriever = None
generator = None
system_ready = False


def initialize_system(pdf_path: str = "data/ibm_10k_2024.pdf"):
    """Initialize RAG system components."""
    global retriever, generator, system_ready

    status_messages = []

    try:
        # Check for pre-built index
        index_manager = VersionedIndexManager("indices")
        latest_version = index_manager.get_latest_version()

        if latest_version:
            status_messages.append(f"Loading index version: {latest_version}")
            faiss_index, chunks, embeddings, metadata = index_manager.load("latest")
            status_messages.append(f"Loaded {len(chunks)} chunks")

            # Initialize retriever with loaded data
            from src.embeddings import EmbeddingModel
            embedding_model = EmbeddingModel()

            # Build retriever (will rebuild indices from chunks)
            retriever = HybridRetriever(
                chunks=chunks,
                embedding_model=embedding_model,
                use_cross_encoder=True
            )
        else:
            # Build from scratch
            status_messages.append("No pre-built index found. Building from PDF...")

            # Load and chunk document
            chunker = DocumentChunker()
            chunks = chunker.process_pdf(pdf_path)
            status_messages.append(f"Chunked document: {len(chunks)} chunks")

            # Build retriever
            retriever = HybridRetriever(chunks=chunks, use_cross_encoder=True)
            status_messages.append("Built retriever with hybrid search + re-ranking")

        # Initialize generator
        generator = AnswerGenerator()
        status_messages.append("Loaded answer generator (FLAN-T5)")

        system_ready = True
        status_messages.append("System ready!")

    except Exception as e:
        status_messages.append(f"Error: {str(e)}")
        system_ready = False

    return "\n".join(status_messages)


def ask_question(
    question: str,
    use_rerank: bool = True,
    use_hybrid: bool = True,
    show_sources: bool = True,
    top_k: int = 5
):
    """
    Process question and return answer with sources.

    Args:
        question: User's question
        use_rerank: Enable cross-encoder re-ranking
        use_hybrid: Use hybrid search (vs dense-only)
        show_sources: Include source excerpts
        top_k: Number of sources to retrieve

    Returns:
        Tuple of (answer, sources_text, metrics_text)
    """
    if not system_ready:
        return (
            "System not initialized. Please click 'Initialize System' first.",
            "",
            ""
        )

    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        start_time = time.time()

        # Retrieve relevant chunks
        if use_hybrid:
            results, metrics = retriever.search(
                query=question,
                top_k=top_k,
                use_rerank=use_rerank,
                apply_section_boost=True
            )
        else:
            results = retriever.baseline_search(question, top_k=top_k)
            metrics = {'method': 'dense_only'}

        retrieval_time = (time.time() - start_time) * 1000

        # Generate answer
        gen_start = time.time()
        response = generator.generate_with_sources(question, results)
        gen_time = (time.time() - gen_start) * 1000

        answer = response['answer']

        # Format sources
        sources_text = ""
        if show_sources:
            sources_text = format_sources(results)

        # Format metrics
        total_time = retrieval_time + gen_time
        metrics_text = format_metrics(metrics, retrieval_time, gen_time, total_time)

        return answer, sources_text, metrics_text

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def format_sources(results):
    """Format source chunks for display."""
    lines = ["### Retrieved Sources\n"]

    for i, (chunk, score) in enumerate(results, 1):
        section = chunk.get('section', 'unknown').replace('_', ' ').title()
        preview = chunk['text'][:300].replace('\n', ' ')
        if len(chunk['text']) > 300:
            preview += "..."

        lines.append(f"**Source {i}** (Section: {section}, Score: {score:.3f})")
        lines.append(f"> {preview}\n")

    return "\n".join(lines)


def format_metrics(metrics, retrieval_ms, gen_ms, total_ms):
    """Format performance metrics for display."""
    lines = ["### Performance Metrics\n"]

    method = metrics.get('method', 'hybrid')
    lines.append(f"- **Search Method**: {method}")

    if metrics.get('reranked'):
        lines.append("- **Re-ranking**: Enabled (cross-encoder)")
    else:
        lines.append("- **Re-ranking**: Disabled")

    if metrics.get('section_boosted'):
        lines.append("- **Section Boost**: Enabled")

    lines.append(f"- **Retrieval Time**: {retrieval_ms:.1f}ms")
    lines.append(f"- **Generation Time**: {gen_ms:.1f}ms")
    lines.append(f"- **Total Time**: {total_ms:.1f}ms")

    return "\n".join(lines)


# Example questions for the UI
EXAMPLE_QUESTIONS = [
    "What intellectual property risks does IBM face?",
    "What are IBM's main revenue segments?",
    "How is IBM positioning itself in the hybrid cloud market?",
    "What is IBM's research and development spending?",
    "What cybersecurity risks does IBM identify?",
    "How does IBM describe its business strategy?",
    "What is IBM's free cash flow?",
    "What quantum computing initiatives does IBM have?",
]


def create_interface():
    """Create Gradio interface."""

    with gr.Blocks(
        title="IBM 10-K RAG System",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # IBM 10-K RAG System

        **Production-grade retrieval-augmented generation for SEC filing analysis**

        This system uses hybrid search (BM25 + dense embeddings), cross-encoder re-ranking,
        and local LLM generation to answer questions about IBM's 2024 10-K filing.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Main Q&A interface
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about IBM's risks, revenue, strategy, technology...",
                    lines=2
                )

                with gr.Row():
                    submit_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("Clear")

                answer_output = gr.Textbox(
                    label="Answer",
                    lines=6,
                    interactive=False
                )

                sources_output = gr.Markdown(label="Sources")

            with gr.Column(scale=1):
                # Configuration options
                gr.Markdown("### Configuration")

                use_rerank = gr.Checkbox(
                    label="Cross-Encoder Re-ranking",
                    value=True,
                    info="More accurate but slower"
                )

                use_hybrid = gr.Checkbox(
                    label="Hybrid Search (BM25 + Dense)",
                    value=True,
                    info="Better for keyword-heavy queries"
                )

                show_sources = gr.Checkbox(
                    label="Show Source Excerpts",
                    value=True
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Sources"
                )

                metrics_output = gr.Markdown(label="Metrics")

                # System status
                gr.Markdown("### System Status")
                init_btn = gr.Button("Initialize System", variant="secondary")
                status_output = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False
                )

        # Example questions
        gr.Markdown("### Example Questions")
        with gr.Row():
            for i in range(4):
                gr.Button(
                    EXAMPLE_QUESTIONS[i][:40] + "...",
                    size="sm"
                ).click(
                    lambda q=EXAMPLE_QUESTIONS[i]: q,
                    outputs=[question_input]
                )

        with gr.Row():
            for i in range(4, 8):
                gr.Button(
                    EXAMPLE_QUESTIONS[i][:40] + "...",
                    size="sm"
                ).click(
                    lambda q=EXAMPLE_QUESTIONS[i]: q,
                    outputs=[question_input]
                )

        # Event handlers
        submit_btn.click(
            fn=ask_question,
            inputs=[question_input, use_rerank, use_hybrid, show_sources, top_k],
            outputs=[answer_output, sources_output, metrics_output]
        )

        question_input.submit(
            fn=ask_question,
            inputs=[question_input, use_rerank, use_hybrid, show_sources, top_k],
            outputs=[answer_output, sources_output, metrics_output]
        )

        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[question_input, answer_output, sources_output, metrics_output]
        )

        init_btn.click(
            fn=initialize_system,
            outputs=[status_output]
        )

        # Footer
        gr.Markdown("""
        ---
        **Technical Stack**: FAISS + BM25 hybrid search | Cross-encoder re-ranking | FLAN-T5 generation

        Built for portfolio demonstration. Zero API cost through local models.
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
