"""
rag.py - Retrieval-Augmented Generation (RAG) Pipeline with Google Gemini
Part of the RAG-Based Document Question Answering Chatbot project.

Handles:
- ChromaDB semantic retrieval (top-k chunks with metadata and similarity scores)
- Context assembly and prompt engineering
- Google Gemini API integration using the official `google-genai` SDK
- Strict document-grounded system instruction
- Multi-turn conversation support
- Rate-limit handling and graceful error recovery
- Direct LLM (Without RAG) generation for side-by-side comparison
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import Google GenAI SDK
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError:
    genai = None
    genai_errors = None
    types = None

from ingest import VectorStoreManager, get_vector_store

load_dotenv()

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SYSTEM_INSTRUCTION = (
    "You are a document-based AI assistant. Answer questions using only the context "
    "retrieved from the user's uploaded documents. Do not invent information or make "
    "unsupported assumptions. If the answer is not available in the provided context, "
    "clearly state that the uploaded documents do not contain sufficient information. "
    "Provide concise and accurate answers. Mention the source document and page number "
    "whenever possible."
)

DIRECT_SYSTEM_INSTRUCTION = (
    "You are a helpful AI assistant. Answer the user's question directly based on your "
    "general pre-trained knowledge. Do not reference any external uploaded documents "
    "since none are provided."
)


class Retriever:
    """Retrieves the most semantically relevant text chunks from ChromaDB."""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or get_vector_store()

    def retrieve(
        self, query: str, top_k: int = 4, filter_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Embeds the query and searches ChromaDB for the top-k most similar chunks.

        Args:
            query: The user's search or question string.
            top_k: Number of chunks to retrieve (default: 4).
            filter_source: Optional document filename to restrict retrieval to.

        Returns:
            List of retrieved chunk dicts with content, source, page, and similarity.
        """
        if not query or not query.strip():
            return []

        collection = self.vector_store.collection
        count = collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        query_kwargs: Dict[str, Any] = {
            "query_texts": [query.strip()],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_source:
            query_kwargs["where"] = {"source": filter_source}

        try:
            results = collection.query(**query_kwargs)
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

        chunks: List[Dict[str, Any]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return chunks

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            # Chroma with cosine distance: distance is 1 - cosine_similarity
            # Distance near 0 means high similarity; distance near 1 means low similarity.
            similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
            chunks.append(
                {
                    "text": doc_text,
                    "source": meta.get("source", "Unknown Document"),
                    "page": meta.get("page", 1),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": round(float(dist), 4),
                    "similarity": round(similarity_score, 4),
                }
            )

        return chunks


class RAGPipeline:
    """End-to-end RAG pipeline using Google Gemini and ChromaDB."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.vector_store = vector_store or get_vector_store()
        self.retriever = Retriever(vector_store=self.vector_store)
        self._client: Optional[Any] = None

    def get_client(self) -> Any:
        """Lazily instantiates the Google GenAI client."""
        if genai is None:
            raise ImportError(
                "The `google-genai` package is not installed. Please install it with: "
                "pip install google-genai"
            )
        if not self.api_key:
            raise ValueError(
                "Gemini API Key is not set. Please set the GEMINI_API_KEY environment variable "
                "or pass your key in the Streamlit sidebar / .env file."
            )
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def set_api_key(self, api_key: str) -> None:
        """Updates the API key and resets the client."""
        self.api_key = api_key.strip()
        self._client = None

    def set_model(self, model_name: str) -> None:
        """Updates the Gemini model name."""
        self.model_name = model_name.strip()

    @staticmethod
    def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a clean, source-attributed context block."""
        if not retrieved_chunks:
            return "No relevant context found in the uploaded documents."

        formatted_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            src = chunk.get("source", "Unknown")
            page = chunk.get("page", "Unknown")
            sim = chunk.get("similarity", 0.0)
            text = chunk.get("text", "").strip()
            formatted_parts.append(
                f"[Document: {src} | Page: {page} | Relevance Score: {sim:.2f}]\n{text}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    def _call_gemini_with_retry(
        self,
        prompt: str,
        system_instruction: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Executes a Gemini API call with exponential backoff for rate limits (HTTP 429).
        """
        client = self.get_client()

        # Build contents including multi-turn history if provided
        contents_payload: List[Any] = []
        if chat_history:
            for turn in chat_history:
                role = "user" if turn.get("role") == "user" else "model"
                contents_payload.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=turn.get("content", ""))],
                    )
                )

        # Append current user prompt
        contents_payload.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,  # Low temperature for factual precision
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents_payload,
                    config=config,
                )
                if response.text:
                    return response.text.strip()
                return "The model returned an empty response. Please verify the prompt."
            except Exception as e:
                last_error = e
                err_str = str(e)
                # Check for rate limit / 429, 503 high demand, or transient service errors
                is_transient = any(
                    sig in err_str.lower()
                    for sig in [
                        "429",
                        "resourceexhausted",
                        "quota",
                        "503",
                        "unavailable",
                        "high demand",
                        "temporarily unavailable",
                        "deadline exceeded",
                        "timeout",
                    ]
                )
                if is_transient and attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 2
                    print(
                        f"Transient Gemini API status ({err_str[:60]}...). "
                        f"Retrying attempt {attempt + 1}/{max_retries} in {sleep_time}s...",
                        flush=True,
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    # Non-retryable error or exhausted retries
                    raise e
                    # Non-retryable error
                    raise e

        raise RuntimeError(
            f"Gemini API request failed after {max_retries} attempts: {str(last_error)}"
        )

    def answer_question(
        self,
        question: str,
        top_k: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filter_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline:
        1. Retrieve relevant chunks from ChromaDB.
        2. Format context with document metadata.
        3. Send context and question to Google Gemini.
        4. Return answer with citations and retrieval details.
        """
        start_time = time.time()

        # Step 1: Retrieval (Do not generate answer before retrieval completes)
        retrieved_chunks = self.retriever.retrieve(
            query=question, top_k=top_k, filter_source=filter_source
        )

        # Step 2: Check if any documents exist in the database
        stats = self.vector_store.get_statistics()
        if stats["total_chunks"] == 0:
            return {
                "answer": (
                    "No documents have been uploaded or indexed yet. "
                    "Please upload at least one PDF in the sidebar before asking questions."
                ),
                "retrieved_chunks": [],
                "context_used": "",
                "model_used": self.model_name,
                "latency_seconds": round(time.time() - start_time, 2),
                "has_sources": False,
            }

        # Step 3: Handle empty context or low similarity
        if not retrieved_chunks:
            return {
                "answer": (
                    "The uploaded documents do not contain sufficient information "
                    "to answer this question."
                ),
                "retrieved_chunks": [],
                "context_used": "",
                "model_used": self.model_name,
                "latency_seconds": round(time.time() - start_time, 2),
                "has_sources": False,
            }

        # Step 4: Assemble prompt with context
        context_block = self.format_context(retrieved_chunks)
        augmented_prompt = (
            f"Context from uploaded documents:\n"
            f"================================\n"
            f"{context_block}\n"
            f"================================\n\n"
            f"User Question: {question}\n\n"
            f"Instructions: Answer the question using ONLY the context provided above. "
            f"Cite the source document and page number in your answer whenever possible. "
            f"If the answer cannot be found in the context, explicitly say that the uploaded "
            f"documents do not contain sufficient information."
        )

        # Step 5: Generate answer via Gemini
        try:
            answer_text = self._call_gemini_with_retry(
                prompt=augmented_prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                chat_history=chat_history,
            )
        except Exception as e:
            answer_text = f"Error generating answer: {str(e)}"

        elapsed = round(time.time() - start_time, 2)
        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
            "context_used": context_block,
            "model_used": self.model_name,
            "latency_seconds": elapsed,
            "has_sources": len(retrieved_chunks) > 0,
        }

    def generate_direct_answer(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generates an answer directly from Gemini WITHOUT document context (No RAG).
        Used for the compulsory assignment comparison requirement.
        """
        start_time = time.time()
        direct_prompt = (
            f"Question: {question}\n\n"
            f"Please answer this question directly based solely on your pre-trained knowledge."
        )

        try:
            answer_text = self._call_gemini_with_retry(
                prompt=direct_prompt,
                system_instruction=DIRECT_SYSTEM_INSTRUCTION,
                chat_history=chat_history,
            )
        except Exception as e:
            answer_text = f"Error generating direct answer: {str(e)}"

        elapsed = round(time.time() - start_time, 2)
        return {
            "answer": answer_text,
            "model_used": self.model_name,
            "latency_seconds": elapsed,
        }

    def compare_rag_vs_direct(
        self,
        question: str,
        top_k: int = 4,
        filter_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs both RAG and Non-RAG pipelines on the same question to produce
        an authentic side-by-side comparison for the compulsory assignment requirement.
        """
        # 1. Non-RAG generation
        direct_res = self.generate_direct_answer(question=question)

        # 2. RAG generation
        rag_res = self.answer_question(
            question=question, top_k=top_k, filter_source=filter_source
        )

        return {
            "question": question,
            "direct_result": direct_res,
            "rag_result": rag_res,
        }


if __name__ == "__main__":
    import sys

    print("==================================================")
    print("RAG Document Chatbot - Query & Retrieval CLI")
    print("==================================================")
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are RAG-Sequence and RAG-Token models?"
    )
    print(f"Query: {query}\n")

    vstore = get_vector_store()
    retriever = Retriever(vector_store=vstore)
    chunks = retriever.retrieve(query=query, top_k=3)
    print(f"Retrieved {len(chunks)} relevant chunk(s) from ChromaDB:")
    for i, c in enumerate(chunks, 1):
        print(
            f"\n[{i}] Source: {c['source']} (Page {c['page']}) | Cosine Similarity: {c['similarity']:.4f}"
        )
        print(f"    Text excerpt: {c['text'][:250]}...")

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("\nCalling Google Gemini with RAG Pipeline...")
        pipeline = RAGPipeline(api_key=api_key, vector_store=vstore)
        result = pipeline.answer_question(query, top_k=3)
        print("\nAnswer from Gemini:")
        print(result["answer"])
    else:
        print("\nNote: GEMINI_API_KEY not set in .env. Retrieval succeeded; set key to run generation.")
    print("==================================================")

