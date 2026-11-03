"""
app.py - Streamlit Frontend for RAG-Based Document Question Answering Chatbot
Part of the RAG-Based Document Question Answering Chatbot project.

Features:
- Modern dark-themed dashboard layout (slate/teal/cyan accents)
- Multi-PDF upload with page counts, chunk stats, and empty/scanned PDF validation
- Persistent local ChromaDB integration with all-MiniLM-L6-v2 embeddings
- Google Gemini LLM via official google-genai SDK
- Multi-turn conversational chat with source citations & debug expandable retrieval inspector
- Compulsory assignment requirement: Dedicated RAG vs Without RAG comparison section
- Document & Chunk Explorer with semantic search tester
- Real-time document statistics & document deletion/management
"""

import os
import time
from typing import List
import streamlit as st
from dotenv import load_dotenv

# Import project modules
from ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DocumentProcessor,
    VectorStoreManager,
    compute_file_hash,
)
from rag import DEFAULT_GEMINI_MODEL, RAGPipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme CSS
CUSTOM_CSS = """
<style>
/* Main app dark styling */
.stApp {
    background-color: #0b0f17;
    color: #e2e8f0;
}

/* Header & Title Styling */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.main-header h1 {
    color: #38bdf8;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.main-header p {
    color: #94a3b8;
    margin: 4px 0 0 0;
    font-size: 0.95rem;
}

/* Metric / Stat Card */
.stat-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.stat-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}

.stat-num {
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}

.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Citation pills */
.citation-badge {
    display: inline-block;
    background-color: #134e4a;
    color: #2dd4bf;
    border: 1px solid #0d9488;
    padding: 3px 10px;
    border-radius: 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}

/* Comparison side-by-side cards */
.compare-card-direct {
    background: #18181b;
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 16px;
    min-height: 240px;
}

.compare-card-rag {
    background: #042f2e;
    border: 1px solid #14b8a6;
    border-radius: 10px;
    padding: 16px;
    min-height: 240px;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
@st.cache_resource
def get_shared_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_shared_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# Sidebar Setup
with st.sidebar:
    st.markdown("### ⚙️ System Settings")

    # API Configuration
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        help="Get a free key from https://aistudio.google.com/app/apikey. If present in .env, it loads automatically.",
    )

    available_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    default_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if default_model not in available_models:
        available_models.insert(0, default_model)

    model_choice = st.selectbox(
        "Gemini Model",
        options=available_models,
        index=available_models.index(default_model) if default_model in available_models else 0,
    )

    if api_key_input:
        st.caption("🟢 API Key is configured")
    else:
        st.warning("⚠️ Enter your Gemini API Key or set GEMINI_API_KEY in `.env`")

    st.markdown("---")

    # Chunking & Retrieval Parameters
    st.markdown("### ✂️ Chunking & Retrieval Config")
    chunk_size = st.slider(
        "Target Chunk Size (tokens)",
        min_value=200,
        max_value=1000,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
        help="Approximate number of tokens per chunk (default ~500).",
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (tokens)",
        min_value=20,
        max_value=250,
        value=DEFAULT_CHUNK_OVERLAP,
        step=10,
        help="Number of tokens overlapping between consecutive chunks (default ~100).",
    )
    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Number of most relevant chunks passed to Gemini for generation.",
    )

    st.markdown("---")

    # PDF Upload Section
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multi-page real PDF documents (minimum 10-20 pages supported).",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index PDFs", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_files = len(uploaded_files)
            indexed_count = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                fname = uploaded_file.name
                status_text.markdown(f"**Processing ({idx + 1}/{total_files}):** `{fname}`...")

                file_bytes = uploaded_file.read()
                file_hash = compute_file_hash(file_bytes)

                # Check if already indexed
                if st.session_state.vector_store.is_document_indexed(fname, file_hash):
                    st.info(f"ℹ️ `{fname}` is already indexed with matching content. Skipping.")
                    progress_bar.progress((idx + 1) / total_files)
                    continue

                try:
                    # 1. Extract text
                    pages_data, summary = DocumentProcessor.extract_text_from_pdf(
                        file_bytes, filename=fname
                    )

                    # 2. Chunk text
                    chunks = DocumentProcessor.chunk_document(
                        pages_data=pages_data,
                        filename=fname,
                        doc_hash=file_hash,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    # 3. Add to ChromaDB
                    added = st.session_state.vector_store.add_document_chunks(
                        chunks=chunks, filename=fname, overwrite=True
                    )

                    # Also save a copy to the local documents/ directory if not present
                    doc_save_path = os.path.join("./documents", fname)
                    if not os.path.exists(doc_save_path):
                        with open(doc_save_path, "wb") as f_out:
                            f_out.write(file_bytes)

                    st.success(
                        f"✅ `{fname}`: {summary['total_pages']} pages, {added} chunks indexed!"
                    )
                    indexed_count += 1
                except Exception as e:
                    st.error(f"❌ Error processing `{fname}`: {str(e)}")

                progress_bar.progress((idx + 1) / total_files)

            status_text.markdown(f"**Completed!** Processed {total_files} file(s).")
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    # Document Management & Database Stats
    st.markdown("### 📊 Document Statistics")
    db_stats = st.session_state.vector_store.get_statistics()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Docs", db_stats.get("total_documents", 0))
    with col2:
        st.metric("Total Chunks", db_stats.get("total_chunks", 0))

    if db_stats.get("documents"):
        with st.expander("📂 Manage Indexed Documents", expanded=False):
            for doc_name, info in db_stats["documents"].items():
                st.write(f"📄 **{doc_name}**")
                st.caption(f"• Pages: {info['pages_count']} | Chunks: {info['chunks']}")
                if st.button(f"🗑️ Delete {doc_name[:15]}...", key=f"del_{doc_name}"):
                    deleted = st.session_state.vector_store.delete_document(doc_name)
                    st.warning(f"Deleted {deleted} chunks from `{doc_name}`.")
                    # remove from documents/ if present
                    local_p = os.path.join("./documents", doc_name)
                    if os.path.exists(local_p):
                        try:
                            os.remove(local_p)
                        except Exception:
                            pass
                    st.rerun()

            st.markdown("---")
            if st.button("⚠️ Clear Entire Vector Store", type="secondary"):
                st.session_state.vector_store.reset_collection()
                st.session_state.messages = []
                st.warning("All indexed documents and chunks have been erased.")
                st.rerun()


# Initialize RAG Pipeline
rag_pipeline = RAGPipeline(
    api_key=api_key_input,
    model_name=model_choice,
    vector_store=st.session_state.vector_store,
)

# Header Banner
st.markdown(
    """
<div class="main-header">
    <h1>📚 RAG-Based Document Question Answering Chatbot</h1>
    <p>Grounded PDF Retrieval-Augmented Generation powered by Sentence Transformers, ChromaDB, and Google Gemini</p>
</div>
""",
    unsafe_allow_html=True,
)

# Top KPI Metric Row
db_info = st.session_state.vector_store.get_statistics()
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-num">{db_info.get('total_documents', 0)}</div>
            <div class="stat-label">Indexed Documents</div>
        </div>""",
        unsafe_allow_html=True,
    )
with mcol2:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-num">{db_info.get('total_chunks', 0)}</div>
            <div class="stat-label">Vector Chunks</div>
        </div>""",
        unsafe_allow_html=True,
    )
with mcol3:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-num">all-MiniLM-L6-v2</div>
            <div class="stat-label">Embedding Model</div>
        </div>""",
        unsafe_allow_html=True,
    )
with mcol4:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-num">{model_choice}</div>
            <div class="stat-label">Active Gemini LLM</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_chat, tab_compare, tab_explorer = st.tabs(
    [
        "💬 RAG Chatbot",
        "⚖️ RAG vs Without RAG Comparison (Compulsory)",
        "📑 Document & Chunk Explorer",
    ]
)

# ---------------------------------------------------------
# TAB 1: RAG CHATBOT
# ---------------------------------------------------------
with tab_chat:
    # Action bar for chat controls
    cbar1, cbar2 = st.columns([6, 1])
    with cbar1:
        # Document filtering optional
        doc_filter = None
        if db_info.get("indexed_files"):
            doc_filter = st.selectbox(
                "Filter retrieval to a specific document (optional):",
                options=["All Indexed Documents"] + db_info["indexed_files"],
                index=0,
            )
            if doc_filter == "All Indexed Documents":
                doc_filter = None
    with cbar2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Welcome message if chat empty
    if not st.session_state.messages:
        if db_info.get("total_chunks", 0) == 0:
            st.info(
                "💡 **Welcome!** To get started, upload one or more multi-page PDF documents in the sidebar "
                "and click **'Process & Index PDFs'**. The documents will be parsed, chunked, and embedded into local ChromaDB."
            )
        else:
            st.success(
                f"✅ **Vector database ready with {db_info.get('total_chunks', 0)} chunks** across "
                f"{db_info.get('total_documents', 0)} document(s). Type a question below to query the documents!"
            )

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If message contains citations and retrieved chunks, display them nicely
            if msg.get("retrieved_chunks"):
                citations = []
                for chunk in msg["retrieved_chunks"]:
                    label = f"📄 {chunk['source']} (Page {chunk['page']}) [Score: {chunk.get('similarity', 0):.2f}]"
                    if label not in citations:
                        citations.append(label)

                if citations:
                    citation_html = " ".join(
                        [f'<span class="citation-badge">{c}</span>' for c in citations]
                    )
                    st.markdown(f"**Sources:** {citation_html}", unsafe_allow_html=True)

                # Debug expandable context
                with st.expander("🔍 View Retrieved Context & Similarity Details", expanded=False):
                    for i, c in enumerate(msg["retrieved_chunks"], 1):
                        st.markdown(
                            f"**Chunk {i}** | Source: `{c['source']}` | Page: **{c['page']}** | "
                            f"Cosine Similarity: **{c.get('similarity', 0):.4f}** (Distance: {c.get('distance', 0):.4f})"
                        )
                        st.code(c["text"], language=None)

            if msg.get("latency"):
                st.caption(f"⚡ Latency: {msg['latency']}s | Model: {model_choice}")

    # Chat Input
    user_prompt = st.chat_input("Ask a question based on your uploaded documents...")
    if user_prompt:
        if not api_key_input:
            st.error(
                "Please configure your Gemini API Key in the sidebar or `.env` before asking questions."
            )
        else:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Generate RAG response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant document chunks and generating answer..."):
                    chat_context_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]

                    result = rag_pipeline.answer_question(
                        question=user_prompt,
                        top_k=top_k,
                        chat_history=chat_context_history,
                        filter_source=doc_filter,
                    )

                    answer_text = result["answer"]
                    retrieved_chunks = result["retrieved_chunks"]
                    latency = result["latency_seconds"]

                    st.markdown(answer_text)

                    # Display citations
                    if retrieved_chunks:
                        citations = []
                        for chunk in retrieved_chunks:
                            label = f"📄 {chunk['source']} (Page {chunk['page']}) [Score: {chunk.get('similarity', 0):.2f}]"
                            if label not in citations:
                                citations.append(label)

                        if citations:
                            citation_html = " ".join(
                                [f'<span class="citation-badge">{c}</span>' for c in citations]
                            )
                            st.markdown(f"**Sources:** {citation_html}", unsafe_allow_html=True)

                        with st.expander(
                            "🔍 View Retrieved Context & Similarity Details", expanded=False
                        ):
                            for i, c in enumerate(retrieved_chunks, 1):
                                st.markdown(
                                    f"**Chunk {i}** | Source: `{c['source']}` | Page: **{c['page']}** | "
                                    f"Cosine Similarity: **{c.get('similarity', 0):.4f}** (Distance: {c.get('distance', 0):.4f})"
                                )
                                st.code(c["text"], language=None)

                    st.caption(f"⚡ Latency: {latency}s | Model: {model_choice}")

                    # Save to session history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "retrieved_chunks": retrieved_chunks,
                            "latency": latency,
                        }
                    )


# ---------------------------------------------------------
# TAB 2: RAG VS WITHOUT RAG COMPARISON (COMPULSORY)
# ---------------------------------------------------------
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Comparison: RAG vs Standard LLM (Without RAG)")
    st.markdown(
        """
    This section is designed to demonstrate the critical difference between a standard pre-trained LLM and a 
    Retrieval-Augmented Generation (RAG) system:
    - **Without RAG (Direct Gemini):** Answers using only general pre-trained weights. When asked about specific, private, 
      recent, or proprietary documents, it either hallucinates plausible-sounding facts or admits it does not know.
    - **With RAG (ChromaDB + Gemini):** Semantically searches the uploaded PDF for the most relevant passages and 
      supplies them as context, forcing the LLM to ground its answer in document facts with citations.
    """
    )

    compare_question = st.text_input(
        "Enter a question based on a specific fact from your uploaded document:",
        placeholder="e.g. What were the specific quarterly metrics reported on page 5?",
    )

    run_comp = st.button("🔬 Run Side-by-Side Comparison Test", type="primary")

    if run_comp:
        if not api_key_input:
            st.error("Please configure your Gemini API Key in the sidebar or `.env` first.")
        elif not compare_question.strip():
            st.warning("Please type a question to test.")
        elif db_info.get("total_chunks", 0) == 0:
            st.error(
                "No documents are indexed. Please upload and index a PDF in the sidebar first."
            )
        else:
            with st.spinner("Executing both pipelines for authentic comparison..."):
                comp_data = rag_pipeline.compare_rag_vs_direct(
                    question=compare_question, top_k=top_k
                )
                st.session_state.comparison_result = comp_data

    # Display comparison if available
    if st.session_state.comparison_result:
        res = st.session_state.comparison_result
        direct = res["direct_result"]
        rag = res["rag_result"]

        st.markdown(f"#### ❓ Question: *\"{res['question']}\"*")

        col_dir, col_rag = st.columns(2)

        with col_dir:
            st.markdown(
                """
            <div style="background-color: #271418; border: 1px solid #f87171; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <h4 style="color: #f87171; margin: 0;">❌ Without RAG (Direct LLM)</h4>
                <small style="color: #fca5a5;">Relies solely on pre-trained parametric weights (No document context)</small>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(direct["answer"])
            st.caption(f"⚡ Latency: {direct['latency_seconds']}s | Model: {direct['model_used']}")

        with col_rag:
            st.markdown(
                """
            <div style="background-color: #063c3b; border: 1px solid #2dd4bf; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <h4 style="color: #2dd4bf; margin: 0;">✅ With RAG Pipeline</h4>
                <small style="color: #99f6e4;">Retrieved from ChromaDB + Grounded Gemini Generation</small>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(rag["answer"])

            # Citations
            if rag.get("retrieved_chunks"):
                citations = []
                for chunk in rag["retrieved_chunks"]:
                    label = f"📄 {chunk['source']} (Page {chunk['page']})"
                    if label not in citations:
                        citations.append(label)
                if citations:
                    citation_html = " ".join(
                        [f'<span class="citation-badge">{c}</span>' for c in citations]
                    )
                    st.markdown(f"**Sources:** {citation_html}", unsafe_allow_html=True)

            st.caption(f"⚡ Latency: {rag['latency_seconds']}s | Model: {rag['model_used']}")

        # Context details used for RAG
        st.markdown("---")
        with st.expander(
            "🔍 Inspect Retrieved Chunks Fed into RAG Pipeline Context", expanded=True
        ):
            if rag.get("retrieved_chunks"):
                for i, chunk in enumerate(rag["retrieved_chunks"], 1):
                    st.markdown(
                        f"**Chunk {i}** | Document: `{chunk['source']}` | Page: **{chunk['page']}** | "
                        f"Relevance: **{chunk.get('similarity', 0):.4f}**"
                    )
                    st.code(chunk["text"], language=None)
            else:
                st.write("No chunks retrieved.")

        # Analysis summary table
        st.markdown("#### 🔬 Pipeline Differences Summary")
        st.markdown(
            """
        | Dimension | Without RAG (Direct Gemini) | With RAG Pipeline (ChromaDB + Gemini) |
        | :--- | :--- | :--- |
        | **Information Source** | Pre-training cutoff data & general web corpus | Uploaded user PDF documents |
        | **Document Specificity** | Cannot access private or new document details | Exact excerpts retrieved semantically |
        | **Hallucination Risk** | **High** (often invents plausible facts) | **Very Low** (strictly restricted by prompt) |
        | **Citations & Pages** | None (cannot cite specific pages) | Direct page & document references |
        | **Verifiability** | Difficult to verify claims | High (user can verify against chunk excerpts) |
        """
        )


# ---------------------------------------------------------
# TAB 3: DOCUMENT & CHUNK EXPLORER
# ---------------------------------------------------------
with tab_explorer:
    st.markdown("### 📑 Document & Chunk Explorer")
    st.markdown("Inspect indexed document pages, chunk distributions, and test vector queries.")

    if db_info.get("total_chunks", 0) == 0:
        st.info("No documents are currently indexed in ChromaDB.")
    else:
        st.markdown("#### 📂 Indexed Files Overview")
        doc_table = []
        for name, dmeta in db_info["documents"].items():
            doc_table.append(
                {
                    "Document Name": name,
                    "Total Pages": dmeta["pages_count"],
                    "Total Chunks": dmeta["chunks"],
                    "Indexed Pages": ", ".join(map(str, dmeta["pages"][:15]))
                    + ("..." if len(dmeta["pages"]) > 15 else ""),
                }
            )
        st.dataframe(doc_table, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Test Semantic Retrieval (Without Calling LLM)")
        test_q = st.text_input(
            "Enter a search query to inspect raw vector similarity scores:",
            placeholder="Search keywords or concepts...",
        )
        test_k = st.slider("Number of chunks to retrieve for inspection", 1, 10, 3)

        if st.button("Search Vector Store"):
            if test_q:
                retriever = rag_pipeline.retriever
                results = retriever.retrieve(query=test_q, top_k=test_k)
                if not results:
                    st.warning("No matches found.")
                else:
                    st.success(f"Retrieved {len(results)} chunks matching '{test_q}':")
                    for i, r in enumerate(results, 1):
                        st.markdown(
                            f"**Match #{i}** — Source: `{r['source']}` | Page: **{r['page']}** | "
                            f"Cosine Similarity: **{r['similarity']:.4f}** (Distance: {r['distance']:.4f})"
                        )
                        st.code(r["text"], language=None)
            else:
                st.warning("Please enter a query first.")
