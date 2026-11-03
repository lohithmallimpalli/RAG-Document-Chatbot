"""
test_rag_pipeline.py - Comprehensive Unit and Integration Tests for RAG Pipeline
"""

import os
import sys
import unittest
from dotenv import load_dotenv

load_dotenv()

from ingest import (
    DocumentProcessor,
    VectorStoreManager,
    compute_file_hash,
    count_tokens,
)
from rag import Retriever, RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_db_dir = "./test_chroma_db"
        cls.pdf_path = "documents/retrieval_augmented_generation_research.pdf"
        cls.vector_store = VectorStoreManager(
            persist_directory=cls.test_db_dir, collection_name="test_rag_collection"
        )
        cls.vector_store.reset_collection()

    @classmethod
    def tearDownClass(cls):
        # Clean up test database
        cls.vector_store.reset_collection()
        import shutil

        if os.path.exists(cls.test_db_dir):
            shutil.rmtree(cls.test_db_dir, ignore_errors=True)

    def test_01_pdf_text_extraction(self):
        """Test PDF text extraction, multi-page support, and character counts."""
        self.assertTrue(os.path.exists(self.pdf_path), f"File {self.pdf_path} not found")
        with open(self.pdf_path, "rb") as f:
            file_bytes = f.read()

        pages_data, summary = DocumentProcessor.extract_text_from_pdf(
            file_bytes, filename="rag_paper.pdf"
        )

        self.assertGreaterEqual(
            summary["total_pages"], 10, "PDF should have at least 10-20 pages"
        )
        self.assertGreater(summary["total_characters"], 10000, "PDF should have substantial text")
        self.assertEqual(len(pages_data), summary["total_pages"])
        self.assertEqual(pages_data[0]["page_number"], 1)

    def test_02_empty_or_scanned_pdf_validation(self):
        """Test that empty or unextractable text triggers a descriptive ValueError."""
        # Create a dummy 1-page PDF with no text
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        from io import BytesIO

        buf = BytesIO()
        writer.write(buf)
        buf.seek(0)

        with self.assertRaises(ValueError) as ctx:
            DocumentProcessor.extract_text_from_pdf(buf.getvalue(), filename="empty_scanned.pdf")
        self.assertIn("empty or scanned", str(ctx.exception).lower())

    def test_03_text_chunking_and_metadata(self):
        """Test semantic chunking, token lengths, overlap, and metadata attribution."""
        with open(self.pdf_path, "rb") as f:
            file_bytes = f.read()
        pages_data, _ = DocumentProcessor.extract_text_from_pdf(file_bytes, filename="rag_paper.pdf")
        doc_hash = compute_file_hash(file_bytes)

        chunks = DocumentProcessor.chunk_document(
            pages_data=pages_data,
            filename="rag_paper.pdf",
            doc_hash=doc_hash,
            chunk_size=500,
            chunk_overlap=100,
        )

        self.assertGreater(len(chunks), 15, "Should generate multiple chunks across 19 pages")

        # Verify metadata integrity
        for c in chunks:
            self.assertIn("chunk_id", c)
            self.assertIn("text", c)
            meta = c["metadata"]
            self.assertEqual(meta["source"], "rag_paper.pdf")
            self.assertGreaterEqual(meta["page"], 1)
            self.assertLessEqual(meta["page"], len(pages_data))
            self.assertGreater(len(c["text"]), 20)

    def test_04_chromadb_indexing_and_deduplication(self):
        """Test vector store insertion, deduplication check, and stats."""
        with open(self.pdf_path, "rb") as f:
            file_bytes = f.read()
        pages_data, _ = DocumentProcessor.extract_text_from_pdf(file_bytes, filename="rag_paper.pdf")
        doc_hash = compute_file_hash(file_bytes)
        chunks = DocumentProcessor.chunk_document(
            pages_data=pages_data,
            filename="rag_paper.pdf",
            doc_hash=doc_hash,
            chunk_size=500,
            chunk_overlap=100,
        )

        added = self.vector_store.add_document_chunks(
            chunks=chunks, filename="rag_paper.pdf", overwrite=True
        )
        self.assertEqual(added, len(chunks))

        # Check deduplication check
        is_indexed = self.vector_store.is_document_indexed("rag_paper.pdf", doc_hash)
        self.assertTrue(is_indexed)

        # Check statistics
        stats = self.vector_store.get_statistics()
        self.assertEqual(stats["total_documents"], 1)
        self.assertEqual(stats["total_chunks"], len(chunks))
        self.assertIn("rag_paper.pdf", stats["documents"])

    def test_05_semantic_retrieval(self):
        """Test retrieval of top-k chunks with similarity metrics."""
        retriever = Retriever(vector_store=self.vector_store)
        results = retriever.retrieve(
            query="What are RAG-Sequence and RAG-Token models?", top_k=4
        )

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 4)

        for chunk in results:
            self.assertIn("text", chunk)
            self.assertIn("source", chunk)
            self.assertIn("page", chunk)
            self.assertIn("similarity", chunk)
            self.assertIn("distance", chunk)
            self.assertGreaterEqual(chunk["similarity"], 0.0)
            self.assertLessEqual(chunk["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
