import os
import time
import json
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Generator

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ==============================================================================
# CONFIG & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="Lens RAG & Vector Index Tester",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAG_ENDPOINT = "https://lbtoopahmulfgffzjumy.supabase.co/functions/v1/lens-rag-endpoint"
SEARCH_ENDPOINT = "https://lbtoopahmulfgffzjumy.supabase.co/functions/v1/vector-index-search"

# ==============================================================================
# HELPERS
# ==============================================================================
def get_default_api_key() -> str:
    """Read API key from st.secrets or environment variable."""
    try:
        if "LENS_API_KEY" in st.secrets:
            return st.secrets["LENS_API_KEY"]
    except Exception:
        pass
    return os.environ.get("LENS_API_KEY", "")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split long text into overlapping character chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += max(1, chunk_size - overlap)
    return chunks


def make_batches(
    chunks: List[Dict[str, Any]],
    max_items: int = 50,
    max_chars: int = 19900
) -> Generator[List[Dict[str, Any]], None, None]:
    """Yield batches respecting max 50 chunks and max 19,900 cumulative chars."""
    cur_batch: List[Dict[str, Any]] = []
    cur_chars = 0
    for c in chunks:
        c_len = len(c.get("text", ""))
        if cur_batch and (len(cur_batch) >= max_items or (cur_chars + c_len) > max_chars):
            yield cur_batch
            cur_batch = []
            cur_chars = 0
        cur_batch.append(c)
        cur_chars += c_len
    if cur_batch:
        yield cur_batch


def safe_post(url: str, api_key: str, payload: dict, auto_retry_429: bool = True) -> tuple[int, dict, dict]:
    """
    POST with error decoding and optional 429 Retry-After handling.
    Returns (status_code, response_json_or_dict, response_headers).
    """
    headers = {
        "x-api-key": api_key.strip(),
        "Content-Type": "application/json",
    }
    while True:
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            try:
                data = res.json()
            except Exception:
                data = {"raw_text": res.text}

            if res.status_code == 429 and auto_retry_429:
                retry_after = int(res.headers.get("Retry-After", 60))
                st.warning(f"⏳ Rate limit hit (HTTP 429). Waiting {retry_after}s as per Retry-After header...")
                time.sleep(retry_after)
                continue

            return res.status_code, data, dict(res.headers)
        except requests.exceptions.RequestException as e:
            return 0, {"error": f"Network/Request error: {str(e)}"}, {}


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("⚡ Lens RAG Console")
    st.caption("Private 768d Vector Index & Semantic Search Tester")

    default_key = get_default_api_key()
    api_key_input = st.text_input(
        "Lens API Key",
        value=default_key,
        type="password",
        help="Reads automatically from st.secrets['LENS_API_KEY']. Override here if needed.",
    )

    if not api_key_input:
        st.error("⚠️ Enter or configure your Lens API key to test calls.")

    st.divider()
    st.markdown("### 📋 Per-Key Default Limits")
    st.markdown(
        """
        - **Embeddings:** `3 / min`, `100 / 3 hrs`
        - **Searches:** `100 / 3 hrs`
        - **Max Payload:** `19,900 chars` / request
        - **Upsert Chunks:** `1 – 50` chunks / batch
        - **Vector Dim:** `768 float32`
        - **top_k:** max 20 (*query*) / 30 (*search*)
        """
    )

    st.divider()
    auto_retry = st.checkbox("Auto-wait on 429 (Retry-After)", value=True)

# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tab_embed, tab_upsert, tab_query, tab_search, tab_batch_workflow = st.tabs([
    "1️⃣ Embed Text",
    "2️⃣ Upsert Chunks",
    "3️⃣ RAG Query",
    "4️⃣ Vector Index Search",
    "📚 Document Ingest Workflow",
])

# ------------------------------------------------------------------------------
# TAB 1: EMBED
# ------------------------------------------------------------------------------
with tab_embed:
    st.subheader("Mode: `embed` (Generate 768d Vector)")
    st.markdown("Generate 768-dimension embeddings directly via VPS embedding infrastructure without writing to an index.")

    embed_text = st.text_area(
        "Text to embed (max 19,900 chars)",
        value="SEBI master circular on margin requirements for derivative contracts in capital markets.",
        height=140,
    )
    st.caption(f"Character count: **{len(embed_text):,}** / 19,900")

    if st.button("Generate Embedding", type="primary", key="btn_embed"):
        if not api_key_input:
            st.error("API key is required.")
        elif not embed_text.strip():
            st.warning("Please enter text.")
        elif len(embed_text) > 19900:
            st.error("Input exceeds the 19,900 character limit.")
        else:
            with st.spinner("Calling embedding engine..."):
                t0 = time.time()
                status, resp, headers = safe_post(
                    RAG_ENDPOINT,
                    api_key_input,
                    {"mode": "embed", "inputs": embed_text},
                    auto_retry_429=auto_retry,
                )
                elapsed = time.time() - t0

            if status == 200 and resp.get("success"):
                vec = resp.get("embedding", [])
                st.success(f"✅ Success ({elapsed:.2f}s) — Dimension: **{len(vec)}**")
                
                # Metrics & Preview
                col1, col2, col3 = st.columns(3)
                col1.metric("Dimension", resp.get("dimension", len(vec)))
                col2.metric("Status", status)
                col3.metric("Latency", f"{elapsed:.2f}s")

                st.markdown("**Vector Preview (first 10 values):**")
                st.code(json.dumps(vec[:10], indent=2), language="json")

                # Store in session state for testing with vector search tab
                st.session_state["last_embedding"] = vec
                st.info("💡 Saved vector in session state. You can test searching with it under Tab 4.")
            else:
                st.error(f"❌ Failed with HTTP {status}")
                st.json(resp)

# ------------------------------------------------------------------------------
# TAB 2: UPSERT CHUNKS
# ------------------------------------------------------------------------------
with tab_upsert:
    st.subheader("Mode: `upsert` (Store Chunks in Private Index)")
    st.markdown(
        "Embeds and saves chunks into your assigned index. Key format: `document_id:chunk_id`. "
        "Re-sending the same keys overwrites existing records."
    )

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        doc_id = st.text_input("Document ID", value="credit-policy-2026", help="Max 120 chars")
    with col_meta2:
        doc_title = st.text_input("Document Title (optional)", value="Comprehensive Credit Risk Policy 2026")

    st.markdown("##### Chunks Editor (Manual Mode)")
    st.caption("Provide up to 50 chunks; cumulative text must be ≤ 19,900 chars.")

    default_chunks = [
        {
            "chunk_id": "c1",
            "page_number": 1,
            "text": "Chapter 1: Loan to Value (LTV) limits for NBFC lending against shares shall not exceed 50% under any circumstance.",
        },
        {
            "chunk_id": "c2",
            "page_number": 2,
            "text": "Chapter 2: Margin calls must be triggered when collateral value falls below 120% of the outstanding loan principal.",
        },
    ]

    chunks_raw = st.text_area(
        "Chunks JSON array",
        value=json.dumps(default_chunks, indent=2),
        height=220,
        help="Array of objects: text (required), chunk_id, page_number, metadata (optional)",
    )

    if st.button("Upsert Chunks", type="primary", key="btn_upsert"):
        if not api_key_input:
            st.error("API key is required.")
        elif not doc_id.strip():
            st.warning("Document ID is required.")
        else:
            try:
                parsed_chunks = json.loads(chunks_raw)
                if not isinstance(parsed_chunks, list) or len(parsed_chunks) == 0:
                    st.error("Chunks must be a non-empty JSON list.")
                    st.stop()
            except json.JSONDecodeError as err:
                st.error(f"Invalid JSON in chunks editor: {err}")
                st.stop()

            total_chars = sum(len(c.get("text", "")) for c in parsed_chunks)
            if len(parsed_chunks) > 50 or total_chars > 19900:
                st.error(f"Batch limits violated: {len(parsed_chunks)} chunks, {total_chars:,} chars. Max is 50 items and 19,900 chars.")
                st.stop()

            payload = {
                "mode": "upsert",
                "document_id": doc_id.strip(),
                "document_title": doc_title.strip() if doc_title else None,
                "chunks": parsed_chunks,
            }

            with st.spinner("Upserting chunks to private vector index..."):
                status, resp, headers = safe_post(
                    RAG_ENDPOINT, api_key_input, payload, auto_retry_429=auto_retry
                )

            if status == 200 and resp.get("success"):
                st.success(f"✅ Successfully upserted {resp.get('upserted', len(parsed_chunks))} chunks!")
                st.json(resp)
            else:
                st.error(f"❌ Upsert failed with HTTP {status}")
                st.json(resp)

# ------------------------------------------------------------------------------
# TAB 3: RAG QUERY
# ------------------------------------------------------------------------------
with tab_query:
    st.subheader("Mode: `query` (Embed Question + Search Private Index)")
    st.markdown("Takes your natural language question, embeds it with the VPS model, and retrieves top matching chunks from your index.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        query_text = st.text_input(
            "Question / Query",
            value="What is the maximum LTV permitted against shares?",
        )
    with col_q2:
        top_k_query = st.slider("top_k", min_value=1, max_value=20, value=5)

    if st.button("Run RAG Query", type="primary", key="btn_rag_query"):
        if not api_key_input:
            st.error("API key is required.")
        elif not query_text.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying private RAG index..."):
                t0 = time.time()
                status, resp, headers = safe_post(
                    RAG_ENDPOINT,
                    api_key_input,
                    {"mode": "query", "inputs": query_text.strip(), "top_k": top_k_query},
                    auto_retry_429=auto_retry,
                )
                elapsed = time.time() - t0

            if status == 200:
                results = resp.get("results") or resp.get("data") or []
                st.success(f"✅ Found {len(results)} matches in {elapsed:.2f}s")

                if results:
                    for i, r in enumerate(results, 1):
                        meta = r.get("metadata", {}) or {}
                        score = r.get("distance", r.get("score", "N/A"))
                        key = r.get("key", f"result_{i}")
                        with st.expander(f"#{i} Match — Key: `{key}` | Distance: `{score}`", expanded=(i == 1)):
                            st.markdown(f"**Stored Chunk Text:**")
                            st.info(meta.get("text", "(No text field in stored metadata)"))
                            st.markdown("**Metadata:**")
                            st.json(meta)
                else:
                    st.warning("No matches returned from your index.")
                    st.json(resp)
            else:
                st.error(f"❌ Query failed with HTTP {status}")
                st.json(resp)

# ------------------------------------------------------------------------------
# TAB 4: VECTOR INDEX SEARCH
# ------------------------------------------------------------------------------
with tab_search:
    st.subheader("Direct Endpoint: `/vector-index-search`")
    st.markdown("Search private vector index via query text or pre-computed 768d vector.")

    search_mode = st.radio("Search input type:", ["Text Query", "Pre-computed 768d Vector"], horizontal=True)

    col_s1, col_s2 = st.columns([3, 1])
    with col_s2:
        top_k_search = st.slider("top_k", min_value=1, max_value=30, value=5, key="slider_s")

    search_payload = {"top_k": top_k_search}

    if search_mode == "Text Query":
        with col_s1:
            s_text = st.text_input("Semantic search text", value="margin call triggers and limits")
        search_payload["query"] = s_text
    else:
        with col_s1:
            has_session_vec = "last_embedding" in st.session_state
            st.caption("Paste 768 floats as JSON array or use vector from Tab 1.")
            default_vec_val = ""
            if has_session_vec:
                default_vec_val = json.dumps(st.session_state["last_embedding"])

            raw_vec_str = st.text_area(
                "queryVector (768 numbers)",
                value=default_vec_val,
                height=130,
                placeholder="[0.0123, -0.0456, ...]",
            )

    if st.button("Search Vector Index", type="primary", key="btn_vector_search"):
        if not api_key_input:
            st.error("API key is required.")
        else:
            if search_mode == "Pre-computed 768d Vector":
                try:
                    vec_parsed = json.loads(raw_vec_str)
                    if not isinstance(vec_parsed, list) or len(vec_parsed) != 768:
                        st.error(f"Vector must contain exactly 768 floats (got {len(vec_parsed)}).")
                        st.stop()
                    search_payload["queryVector"] = vec_parsed
                except Exception as e:
                    st.error(f"Could not parse vector JSON: {e}")
                    st.stop()

            with st.spinner("Executing index search..."):
                t0 = time.time()
                status, resp, headers = safe_post(
                    SEARCH_ENDPOINT, api_key_input, search_payload, auto_retry_429=auto_retry
                )
                elapsed = time.time() - t0

            if status == 200 and resp.get("success"):
                results = resp.get("results", [])
                st.success(f"✅ Found {resp.get('count', len(results))} matches in {elapsed:.2f}s")
                if results:
                    rows = []
                    for r in results:
                        meta = r.get("metadata", {}) or {}
                        rows.append({
                            "Key": r.get("key"),
                            "Distance": r.get("distance"),
                            "Doc Title": meta.get("document_title", "-"),
                            "Page": meta.get("page_number", "-"),
                            "Snippet": (meta.get("text") or "")[:120] + "...",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    with st.expander("Detailed Results (JSON)"):
                        st.json(results)
                else:
                    st.info("No matching records found in this index.")
            else:
                st.error(f"❌ Search failed with HTTP {status}")
                st.json(resp)

# ------------------------------------------------------------------------------
# TAB 5: AUTOMATED DOCUMENT INGEST WORKFLOW
# ------------------------------------------------------------------------------
with tab_batch_workflow:
    st.subheader("📚 Automated Document Ingest (PDF / Text)")
    st.markdown(
        "Upload a document or paste full text. The app automatically splits it into chunks, "
        "bundles them into valid batches (≤ 50 items, ≤ 19,900 chars), and uploads with automatic rate-limit backoff."
    )

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        wf_doc_id = st.text_input("Workflow Document ID", value="annual-report-2025")
    with col_w2:
        wf_doc_title = st.text_input("Workflow Document Title", value="FY25 Annual Audit Report")

    chunk_size = st.number_input("Target Chunk Size (characters)", min_value=200, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=500, value=150, step=25)

    input_source = st.radio("Source:", ["Paste Raw Text", "Upload Text File", "Upload PDF File"], horizontal=True)

    extracted_chunks: List[Dict[str, Any]] = []

    if input_source == "Paste Raw Text":
        pasted = st.text_area("Document Content", height=200)
        if pasted.strip():
            raw_splits = chunk_text(pasted, chunk_size=int(chunk_size), overlap=int(chunk_overlap))
            extracted_chunks = [
                {"chunk_id": f"chunk_{i+1}", "page_number": 1, "text": c}
                for i, c in enumerate(raw_splits)
            ]

    elif input_source == "Upload Text File":
        uploaded_txt = st.file_uploader("Upload .txt file", type=["txt", "md"])
        if uploaded_txt is not None:
            text_content = uploaded_txt.read().decode("utf-8", errors="ignore")
            raw_splits = chunk_text(text_content, chunk_size=int(chunk_size), overlap=int(chunk_overlap))
            extracted_chunks = [
                {"chunk_id": f"chunk_{i+1}", "page_number": 1, "text": c}
                for i, c in enumerate(raw_splits)
            ]

    elif input_source == "Upload PDF File":
        if not PDF_SUPPORT:
            st.error("pypdf is not installed. Run `pip install pypdf`.")
        else:
            uploaded_pdf = st.file_uploader("Upload .pdf", type=["pdf"])
            if uploaded_pdf is not None:
                reader = PdfReader(uploaded_pdf)
                pdf_chunks = []
                c_counter = 1
                for page_idx, page in enumerate(reader.pages):
                    ptxt = page.extract_text() or ""
                    splits = chunk_text(ptxt, chunk_size=int(chunk_size), overlap=int(chunk_overlap))
                    for s in splits:
                        pdf_chunks.append({
                            "chunk_id": f"p{page_idx+1}_c{c_counter}",
                            "page_number": page_idx + 1,
                            "text": s,
                        })
                        c_counter += 1
                extracted_chunks = pdf_chunks

    if extracted_chunks:
        st.success(f"Parsed **{len(extracted_chunks)}** total chunks.")
        batches = list(make_batches(extracted_chunks, max_items=50, max_chars=19900))
        st.info(f"Batched into **{len(batches)}** API request(s) respecting the 50-chunk and 19.9k-char thresholds.")

        with st.expander("Preview First 3 Chunks"):
            st.json(extracted_chunks[:3])

        if st.button("🚀 Upload All Batches to Index", type="primary"):
            if not api_key_input:
                st.error("API key is required.")
                st.stop()
            if not wf_doc_id.strip():
                st.error("Document ID is required.")
                st.stop()

            progress_bar = st.progress(0.0)
            status_box = st.empty()
            total_upserted = 0

            for idx, batch in enumerate(batches):
                status_box.markdown(f"**Uploading batch {idx+1}/{len(batches)} ({len(batch)} chunks)...**")
                payload = {
                    "mode": "upsert",
                    "document_id": wf_doc_id.strip(),
                    "document_title": wf_doc_title.strip() if wf_doc_title else None,
                    "chunks": batch,
                }
                status, resp, headers = safe_post(
                    RAG_ENDPOINT, api_key_input, payload, auto_retry_429=True
                )
                if status == 200 and resp.get("success"):
                    total_upserted += resp.get("upserted", len(batch))
                else:
                    st.error(f"Failed at batch {idx+1}: HTTP {status} — {resp}")
                    st.stop()

                progress_bar.progress((idx + 1) / len(batches))
                # Add slight delay between batches to stay within 3 req/min if not retried
                time.sleep(1.0)

            status_box.empty()
            st.balloons()
            st.success(f"🎉 Successfully ingested entire document: {total_upserted} chunks indexed across {len(batches)} batches!")
