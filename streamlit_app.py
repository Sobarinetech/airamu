import os
import time
import requests
import streamlit as st

# ==============================================================================
# CONFIG & ENDPOINT
# ==============================================================================
st.set_page_config(page_title="Lens RAG Tester", page_icon="⚡", layout="centered")

API_URL = "https://lbtoopahmulfgffzjumy.supabase.co/functions/v1/lens-rag-endpoint"

# Read API key from secrets or environment
API_KEY = st.secrets.get("LENS_API_KEY", os.environ.get("LENS_API_KEY", ""))

st.title("⚡ Lens RAG — Index & Query")

# Sidebar for API key
with st.sidebar:
    st.subheader("API Configuration")
    api_key = st.text_input("Lens API Key", value=API_KEY, type="password")
    batch_size = st.slider("Chunks per request", min_value=1, max_value=10, value=3,
                           help="Keep at 3-5 to prevent VPS 502 overload from simultaneous embeddings.")

if not api_key:
    st.error("⚠️ Enter your Lens API Key in the sidebar or in .streamlit/secrets.toml")
    st.stop()

# ==============================================================================
# HELPERS
# ==============================================================================
def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100):
    """Splits text into manageable chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start += max(1, chunk_size - overlap)
    return chunks


def send_upsert_batch(api_key: str, doc_id: str, chunk_items: list, max_retries: int = 4):
    """Sends a batch of chunks with retries for 429 rate-limits or 502 VPS glitches."""
    headers = {"x-api-key": api_key.strip(), "Content-Type": "application/json"}
    payload = {
        "mode": "upsert",
        "document_id": doc_id,
        "chunks": chunk_items,
    }

    for attempt in range(max_retries):
        try:
            res = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                return True, res.json()
            elif res.status_code == 429:
                wait_time = int(res.headers.get("Retry-After", 20))
                time.sleep(wait_time)
                continue
            elif res.status_code == 502 or "502" in res.text:
                time.sleep(3 * (attempt + 1))  # backoff for VPS worker
                continue
            else:
                return False, res.json() if res.headers.get("content-type") == "application/json" else res.text
        except Exception as e:
            if attempt == max_retries - 1:
                return False, str(e)
            time.sleep(2)
    return False, "Exceeded retry attempts (502 / upstream timeout)"


# ==============================================================================
# TABS: INGEST & QUERY
# ==============================================================================
tab_ingest, tab_query = st.tabs(["📥 1. Paste & Ingest", "🔍 2. Search & Query"])

# ------------------------------------------------------------------------------
# 1. INGEST
# ------------------------------------------------------------------------------
with tab_ingest:
    st.subheader("Ingest Text into Vector Index")

    doc_id = st.text_input("Document ID / Tag", value="doc-001")
    text_content = st.text_area("Paste document / policy text here", height=240, placeholder="Paste your text...")

    if st.button("Chunk & Embed into Index", type="primary"):
        if not text_content.strip():
            st.warning("Please paste some text first.")
        else:
            raw_chunks = split_into_chunks(text_content, chunk_size=800, overlap=100)
            st.write(f"Created **{len(raw_chunks)}** chunks.")

            # Prepare chunk objects
            formatted_chunks = [
                {"chunk_id": f"c_{i+1}", "page_number": 1, "text": c}
                for i, c in enumerate(raw_chunks)
            ]

            # Split into small batches (e.g. 3 at a time)
            batches = [
                formatted_chunks[i : i + batch_size]
                for i in range(0, len(formatted_chunks), batch_size)
            ]

            progress_bar = st.progress(0.0)
            status_box = st.empty()
            total_upserted = 0
            has_error = False

            for idx, batch in enumerate(batches):
                status_box.info(f"Embedding batch {idx + 1} of {len(batches)} ({len(batch)} chunks)...")
                success, resp = send_upsert_batch(api_key, doc_id, batch)

                if success:
                    total_upserted += resp.get("upserted", len(batch))
                    progress_bar.progress((idx + 1) / len(batches))
                    time.sleep(0.5)  # brief pause between calls
                else:
                    st.error(f"Failed on batch {idx + 1}: {resp}")
                    has_error = True
                    break

            status_box.empty()
            if not has_error:
                st.success(f"🎉 Done! Ingested {total_upserted} chunks into your index.")

# ------------------------------------------------------------------------------
# 2. QUERY / SEARCH
# ------------------------------------------------------------------------------
with tab_query:
    st.subheader("Query Your Index")

    query_input = st.text_input("Question / Search Query", placeholder="What are the requirements?")
    top_k = st.slider("Number of results (top_k)", 1, 15, 5)

    if st.button("Run Query", type="primary"):
        if not query_input.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching index..."):
                headers = {"x-api-key": api_key.strip(), "Content-Type": "application/json"}
                payload = {
                    "mode": "query",
                    "inputs": query_input.strip(),
                    "top_k": top_k,
                }
                res = requests.post(API_URL, headers=headers, json=payload, timeout=60)

                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])
                    st.success(f"Found {len(results)} matches:")

                    for i, r in enumerate(results, 1):
                        meta = r.get("metadata", {})
                        chunk_text = meta.get("text", "(No text)")
                        dist = r.get("distance", "N/A")
                        doc = meta.get("document_id", "-")

                        with st.expander(f"Match #{i} — Doc: {doc} | Distance: {dist}", expanded=(i == 1)):
                            st.write(chunk_text)
                            if meta:
                                st.caption(f"Metadata: {meta}")
                else:
                    st.error(f"Query failed with HTTP {res.status_code}: {res.text}")
