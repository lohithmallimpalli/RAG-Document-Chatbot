"""
ingest.py - Document Ingestion, Text Extraction, Chunking, and ChromaDB Indexing
Part of the RAG-Based Document Question Answering Chatbot project.

Handles:
- PDF loading and text extraction via PyPDF
- Multi-page document handling and empty/scanned PDF detection
- Semantic/token-aware text chunking (~500 tokens with ~100 tokens overlap)
- Source filename, page number, and chunk metadata tracking
- Duplicate and empty chunk elimination
- Local persistent ChromaDB vector storage with Sentence Transformers (all-MiniLM-L6-v2)
- Document management (delete, re-index, statistics, change detection)
"""

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple



import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

# Robust and instant token estimation (1 token ~= 0.75 words / ~4 characters)
def count_tokens(text: str) -> int:
    """Estimates token count reliably without network dependencies."""
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) / 0.75))




load_dotenv()

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
COLLECTION_NAME = "rag_documents"


def compute_file_hash(file_bytes_or_path: Any) -> str:
    """Computes SHA-256 hash of a file or bytes to detect modifications."""
    hasher = hashlib.sha256()
    if isinstance(file_bytes_or_path, (str, os.PathLike)):
        with open(file_bytes_or_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(file_bytes_or_path, bytes):
        hasher.update(file_bytes_or_path)
    elif hasattr(file_bytes_or_path, "getvalue"):  # e.g., Streamlit UploadedFile
        hasher.update(file_bytes_or_path.getvalue())
    elif hasattr(file_bytes_or_path, "read"):
        pos = file_bytes_or_path.tell() if hasattr(file_bytes_or_path, "tell") else None
        hasher.update(file_bytes_or_path.read())
        if pos is not None and hasattr(file_bytes_or_path, "seek"):
            file_bytes_or_path.seek(pos)
    else:
        hasher.update(str(file_bytes_or_path).encode("utf-8"))
    return hasher.hexdigest()


class DocumentProcessor:
    """Handles PDF reading, text extraction, validation, and chunking."""

    @staticmethod
    def extract_text_from_pdf(
        file_source: Any, filename: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extracts text from each page of a PDF file using PyPDF.

        Args:
            file_source: File path, file-like object, or bytes of the PDF.
            filename: Human-readable name of the document.

        Returns:
            Tuple of (pages_data, metadata_summary):
            - pages_data: List of dicts: [{"page_number": int, "text": str, "char_count": int}]
            - metadata_summary: {"filename": str, "total_pages": int, "total_characters": int, "status": str}

        Raises:
            ValueError: If the file is not a valid PDF or is completely empty/scanned.
        """
        try:
            import io
            if isinstance(file_source, bytes):
                stream = io.BytesIO(file_source)
            elif hasattr(file_source, "getvalue"):  # Streamlit UploadedFile
                stream = io.BytesIO(file_source.getvalue())
            elif isinstance(file_source, (str, os.PathLike)):
                stream = str(file_source)
            else:
                stream = file_source

            reader = PdfReader(stream)
        except Exception as e:
            raise ValueError(f"Failed to open '{filename}' as a PDF: {str(e)}")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError(f"Document '{filename}' contains 0 pages.")

        pages_data: List[Dict[str, Any]] = []
        total_characters = 0

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                page_text = ""

            # Normalize whitespace
            clean_text = re.sub(r"[ \t]+", " ", page_text).strip()
            char_count = len(clean_text)
            total_characters += char_count

            pages_data.append(
                {
                    "page_number": page_number,
                    "text": clean_text,
                    "char_count": char_count,
                }
            )

        # Scanned PDF or empty check
        if total_characters < 50:
            raise ValueError(
                f"Document '{filename}' appears to be empty or scanned without an OCR text layer "
                f"(extracted only {total_characters} characters across {total_pages} pages). "
                "Please upload a digital PDF with searchable text or perform OCR first."
            )

        summary = {
            "filename": filename,
            "total_pages": total_pages,
            "total_characters": total_characters,
            "status": "valid",
        }
        return pages_data, summary

    @staticmethod
    def chunk_document(
        pages_data: List[Dict[str, Any]],
        filename: str,
        doc_hash: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[Dict[str, Any]]:
        """
        Splits extracted pages into text chunks with metadata.

        Preserves paragraph and sentence boundaries wherever possible.
        Maintains chunk_size and chunk_overlap (in token counts).
        Ensures source filename and page numbers are stored in metadata.
        Avoids duplicate and empty chunks.

        Returns:
            List of chunk dicts:
            [{
                "chunk_id": str,
                "text": str,
                "metadata": {
                    "source": str,
                    "page": int,
                    "chunk_index": int,
                    "doc_hash": str,
                    "token_count": int
                }
            }]
        """
        chunks: List[Dict[str, Any]] = []
        seen_chunks = set()
        global_chunk_idx = 0

        for page_info in pages_data:
            page_num = page_info["page_number"]
            page_text = page_info["text"]

            if not page_text or len(page_text.strip()) == 0:
                continue

            # Split page text into paragraphs first
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in page_text.split("\n") if p.strip()]

            current_chunk_tokens: List[str] = []
            current_token_count = 0

            # Break into sentences or smaller units
            units: List[str] = []
            for p in paragraphs:
                # Split by sentence boundaries (.!? followed by space or newline)
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean:
                        units.append(s_clean)

            # Accumulate units into chunks
            i = 0
            while i < len(units):
                unit = units[i]
                unit_tokens = count_tokens(unit)

                # If a single sentence exceeds chunk size, split it into smaller token slices
                if unit_tokens > chunk_size:
                    words = unit.split()
                    slice_words = []
                    for w in words:
                        slice_words.append(w)
                        if count_tokens(" ".join(slice_words)) >= chunk_size:
                            chunk_str = " ".join(slice_words)
                            chunk_hash = hashlib.md5(chunk_str.encode("utf-8")).hexdigest()
                            if chunk_hash not in seen_chunks:
                                seen_chunks.add(chunk_hash)
                                chunks.append(
                                    {
                                        "chunk_id": f"{filename}_p{page_num}_c{global_chunk_idx}",
                                        "text": chunk_str,
                                        "metadata": {
                                            "source": filename,
                                            "page": page_num,
                                            "chunk_index": global_chunk_idx,
                                            "doc_hash": doc_hash,
                                            "token_count": count_tokens(chunk_str),
                                        },
                                    }
                                )
                                global_chunk_idx += 1
                            slice_words = slice_words[-max(1, int(chunk_overlap * 0.75)) :]
                    i += 1
                    continue

                if current_token_count + unit_tokens <= chunk_size:
                    current_chunk_tokens.append(unit)
                    current_token_count += unit_tokens
                    i += 1
                else:
                    # Flush current chunk
                    if current_chunk_tokens:
                        chunk_str = " ".join(current_chunk_tokens).strip()
                        chunk_hash = hashlib.md5(chunk_str.encode("utf-8")).hexdigest()
                        if chunk_hash not in seen_chunks and len(chunk_str) > 20:
                            seen_chunks.add(chunk_hash)
                            chunks.append(
                                {
                                    "chunk_id": f"{filename}_p{page_num}_c{global_chunk_idx}",
                                    "text": chunk_str,
                                    "metadata": {
                                        "source": filename,
                                        "page": page_num,
                                        "chunk_index": global_chunk_idx,
                                        "doc_hash": doc_hash,
                                        "token_count": current_token_count,
                                    },
                                }
                            )
                            global_chunk_idx += 1

                    # Handle overlap by stepping back some units if possible
                    overlap_tokens = 0
                    overlap_units: List[str] = []
                    for back_unit in reversed(current_chunk_tokens):
                        b_tok = count_tokens(back_unit)
                        if overlap_tokens + b_tok <= chunk_overlap:
                            overlap_units.insert(0, back_unit)
                            overlap_tokens += b_tok
                        else:
                            break

                    current_chunk_tokens = list(overlap_units)
                    current_token_count = overlap_tokens

            # Flush remaining tokens on the page
            if current_chunk_tokens:
                chunk_str = " ".join(current_chunk_tokens).strip()
                chunk_hash = hashlib.md5(chunk_str.encode("utf-8")).hexdigest()
                if chunk_hash not in seen_chunks and len(chunk_str) > 20:
                    seen_chunks.add(chunk_hash)
                    chunks.append(
                        {
                            "chunk_id": f"{filename}_p{page_num}_c{global_chunk_idx}",
                            "text": chunk_str,
                            "metadata": {
                                "source": filename,
                                "page": page_num,
                                "chunk_index": global_chunk_idx,
                                "doc_hash": doc_hash,
                                "token_count": count_tokens(chunk_str),
                            },
                        }
                    )
                    global_chunk_idx += 1

        return chunks


class VectorStoreManager:
    """Manages persistent ChromaDB vector store using Sentence Transformers."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Initialize Sentence Transformers embedding function
        # This guarantees identical embeddings for both chunk indexing and user querying
        self.embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def is_document_indexed(self, filename: str, doc_hash: Optional[str] = None) -> bool:
        """
        Checks whether a document is already indexed in ChromaDB.
        If doc_hash is provided, checks if the indexed document matches the hash.
        """
        try:
            results = self.collection.get(
                where={"source": filename},
                limit=1,
                include=["metadatas"],
            )
            if not results["ids"]:
                return False

            if doc_hash is not None and results["metadatas"]:
                stored_hash = results["metadatas"][0].get("doc_hash")
                return stored_hash == doc_hash

            return True
        except Exception:
            return False

    def add_document_chunks(
        self, chunks: List[Dict[str, Any]], filename: str, overwrite: bool = False
    ) -> int:
        """
        Adds text chunks to ChromaDB.

        Args:
            chunks: List of chunk dictionaries.
            filename: Name of the document.
            overwrite: If True, deletes existing chunks for this document first.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        if overwrite:
            self.delete_document(filename)

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # ChromaDB allows batch insertion; perform in safe batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        return len(chunks)

    def delete_document(self, filename: str) -> int:
        """Deletes all chunks associated with a specific document source."""
        try:
            existing = self.collection.get(where={"source": filename})
            count = len(existing["ids"])
            if count > 0:
                self.collection.delete(where={"source": filename})
            return count
        except Exception as e:
            print(f"Error deleting document '{filename}': {e}")
            return 0

    def reset_collection(self) -> None:
        """Deletes all documents and resets the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics on the vector store."""
        try:
            total_chunks = self.collection.count()
            if total_chunks == 0:
                return {
                    "total_chunks": 0,
                    "total_documents": 0,
                    "documents": {},
                    "indexed_files": [],
                }

            all_data = self.collection.get(include=["metadatas"])
            docs_summary: Dict[str, Dict[str, Any]] = {}

            for meta in all_data["metadatas"]:
                src = meta.get("source", "unknown")
                page = meta.get("page", 1)
                if src not in docs_summary:
                    docs_summary[src] = {"chunk_count": 0, "pages": set()}
                docs_summary[src]["chunk_count"] += 1
                docs_summary[src]["pages"].add(page)

            formatted_docs = {}
            for src, info in docs_summary.items():
                formatted_docs[src] = {
                    "chunks": info["chunk_count"],
                    "pages_count": len(info["pages"]),
                    "pages": sorted(list(info["pages"])),
                }

            return {
                "total_chunks": total_chunks,
                "total_documents": len(formatted_docs),
                "documents": formatted_docs,
                "indexed_files": list(formatted_docs.keys()),
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "documents": {},
                "indexed_files": [],
                "error": str(e),
            }


# Module level convenience functions
def get_vector_store() -> VectorStoreManager:
    """Returns a singleton instance or freshly initialized VectorStoreManager."""
    return VectorStoreManager()


if __name__ == "__main__":
    import pprint

    print("==================================================")
    print("RAG Document Chatbot - Document Ingestion CLI")
    print("==================================================")
    vstore = get_vector_store()
    docs_dir = "./documents"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}. Place PDF documents there to ingest.")
    else:
        print(f"Found {len(pdf_files)} PDF file(s) in '{docs_dir}':\n")
        for pdf_name in pdf_files:
            file_path = os.path.join(docs_dir, pdf_name)
            print(f"Processing: {pdf_name}...")
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            f_hash = compute_file_hash(file_bytes)
            pages_data, summary = DocumentProcessor.extract_text_from_pdf(
                file_bytes, filename=pdf_name
            )
            print(f"  • Extracted {summary['total_pages']} pages ({summary['total_characters']} characters)")

            chunks = DocumentProcessor.chunk_document(
                pages_data=pages_data,
                filename=pdf_name,
                doc_hash=f_hash,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )
            print(f"  • Generated {len(chunks)} chunks (target {DEFAULT_CHUNK_SIZE} tokens, overlap {DEFAULT_CHUNK_OVERLAP})")

            added = vstore.add_document_chunks(chunks=chunks, filename=pdf_name, overwrite=True)
            print(f"  ✓ Indexed {added} chunks into persistent ChromaDB!\n")

    print("Vector Store Summary Statistics:")
    pprint.pprint(vstore.get_statistics())
    print("==================================================")

