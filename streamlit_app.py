import os
import time
import requests
import streamlit as st

# ==============================================================================
# CONFIG & ENDPOINTS
# ==============================================================================
st.set_page_config(page_title="Lens RAG Tester", page_icon="⚡", layout="centered")

BASE_FN_URL = "https://lbtoopahmulfgffzjumy.supabase.co/functions/v1"
API_URL = f"{BASE_FN_URL}/lens-rag-endpoint"      # private index: upsert + query
HYBRID_URL = f"{BASE_FN_URL}/lens-hybrid-rag"     # hybrid search + list_buckets

# Read API key from secrets or environment
try:
    API_KEY = st.secrets.get("LENS_API_KEY", os.environ.get("LENS_API_KEY", ""))
except Exception:
    API_KEY = os.environ.get("LENS_API_KEY", "")

st.title("⚡ Lens RAG — Index, Query & Hybrid Search")

# Sidebar for API key
with st.sidebar:
    st.subheader("API Configuration")
    api_key = st.text_input("Lens API Key", value=API_KEY, type="password")
    batch_size = st.slider(
        "Chunks per request", min_value=1, max_value=10, value=3,
        help="Keep at 3-5 to prevent VPS 502 overload from simultaneous embeddings.",
    )

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


def make_headers(key: str) -> dict:
    return {"x-api-key": key.strip(), "Content-Type": "application/json"}


def timed_post(url: str, key: str, payload: dict, timeout: int = 60):
    """POST and return (response, elapsed_ms) measured client-side."""
    t0 = time.perf_counter()
    res = requests.post(url, headers=make_headers(key), json=payload, timeout=timeout)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return res, elapsed_ms


def safe_json(res: requests.Response):
    try:
        return res.json()
    except Exception:
        return res.text


def send_upsert_batch(key: str, doc_id: str, chunk_items: list, max_retries: int = 4):
    """
    Sends a batch of chunks with retries for 429 rate-limits or 502 VPS glitches.
    Returns (success, response, elapsed_ms) where elapsed_ms covers the whole call
    including retries/backoff.
    """
    payload = {"mode": "upsert", "document_id": doc_id, "chunks": chunk_items}
    t0 = time.perf_counter()

    def elapsed():
        return round((time.perf_counter() - t0) * 1000, 1)

    for attempt in range(max_retries):
        try:
            res = requests.post(API_URL, headers=make_headers(key), json=payload, timeout=60)
            if res.status_code == 200:
                return True, res.json(), elapsed()
            elif res.status_code == 429:
                time.sleep(int(res.headers.get("Retry-After", 20)))
                continue
            elif res.status_code == 502 or "502" in res.text:
                time.sleep(3 * (attempt + 1))  # backoff for VPS worker
                continue
            else:
                return False, safe_json(res), elapsed()
        except Exception as e:
            if attempt == max_retries - 1:
                return False, str(e), elapsed()
            time.sleep(2)
    return False, "Exceeded retry attempts (502 / upstream timeout)", elapsed()


def extract_buckets(data):
    """
    Best-effort parse of the list_buckets response into {bucket: [indexes]}.
    Handles common shapes: {"buckets": ["a", "b"]} or
    {"buckets": [{"name": "a", "indexes": ["x", "y"]}]}. Returns {} if unknown.
    """
    out = {}
    if not isinstance(data, dict):
        return out
    items = data.get("buckets") or data.get("data") or []
    if not isinstance(items, list):
        return out
    for b in items:
        if isinstance(b, str):
            out[b] = []
        elif isinstance(b, dict):
            name = b.get("name") or b.get("bucket") or b.get("id") or b.get("bucket_name")
            if not name:
                continue
            idxs = b.get("indexes") or b.get("indices") or b.get("index_names") or []
            clean = []
            for i in idxs if isinstance(idxs, list) else []:
                if isinstance(i, str):
                    clean.append(i)
                elif isinstance(i, dict):
                    n = i.get("name") or i.get("index") or i.get("id")
                    if n:
                        clean.append(n)
            out[name] = clean
    return out


def show_latency(client_ms, server_ms=None):
    cols = st.columns(2)
    cols[0].metric("Round-trip (client)", f"{client_ms} ms")
    if server_ms is not None:
        cols[1].metric("Server latency", f"{server_ms} ms")


# ==============================================================================
# TABS
# ==============================================================================
tab_ingest, tab_query, tab_hybrid, tab_buckets = st.tabs(
    ["📥 1. Paste & Ingest", "🔍 2. Private Query", "🧬 3. Hybrid Search", "🪣 4. Buckets"]
)

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
            overall_t0 = time.perf_counter()
            raw_chunks = split_into_chunks(text_content, chunk_size=800, overlap=100)
            st.write(f"Created **{len(raw_chunks)}** chunks.")

            formatted_chunks = [
                {"chunk_id": f"c_{i+1}", "page_number": 1, "text": c}
                for i, c in enumerate(raw_chunks)
            ]
            batches = [
                formatted_chunks[i: i + batch_size]
                for i in range(0, len(formatted_chunks), batch_size)
            ]

            progress_bar = st.progress(0.0)
            status_box = st.empty()
            total_upserted = 0
            batch_times = []
            has_error = False

            for idx, batch in enumerate(batches):
                status_box.info(f"Embedding batch {idx + 1} of {len(batches)} ({len(batch)} chunks)...")
                success, resp, ms = send_upsert_batch(api_key, doc_id, batch)
                batch_times.append(ms)

                if success:
                    total_upserted += resp.get("upserted", len(batch))
                    progress_bar.progress((idx + 1) / len(batches))
                    time.sleep(0.5)  # brief pause between calls
                else:
                    st.error(f"Failed on batch {idx + 1} ({ms} ms): {resp}")
                    has_error = True
                    break

            status_box.empty()
            total_ms = round((time.perf_counter() - overall_t0) * 1000, 1)

            if not has_error:
                st.success(f"🎉 Done! Ingested {total_upserted} chunks into your index.")

            if batch_times:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total time", f"{total_ms} ms")
                c2.metric("Avg / batch", f"{round(sum(batch_times) / len(batch_times), 1)} ms")
                c3.metric("Slowest batch", f"{max(batch_times)} ms")
                with st.expander("Per-batch timings"):
                    for i, ms in enumerate(batch_times, 1):
                        st.write(f"Batch {i}: {ms} ms")

# ------------------------------------------------------------------------------
# 2. PRIVATE QUERY (original endpoint)
# ------------------------------------------------------------------------------
with tab_query:
    st.subheader("Query Your Private Index")

    query_input = st.text_input("Question / Search Query", placeholder="What are the requirements?", key="pq")
    top_k = st.slider("Number of results (top_k)", 1, 15, 5, key="pq_topk")

    if st.button("Run Query", type="primary"):
        if not query_input.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching index..."):
                try:
                    payload = {"mode": "query", "inputs": query_input.strip(), "top_k": top_k}
                    res, ms = timed_post(API_URL, api_key, payload)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            if res.status_code == 200:
                data = res.json()
                results = data.get("results", [])
                st.success(f"Found {len(results)} matches:")
                show_latency(ms, data.get("latency_ms"))

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
                st.error(f"Query failed with HTTP {res.status_code} ({ms} ms): {res.text}")

# ------------------------------------------------------------------------------
# 3. HYBRID SEARCH (private 768d + Invision bucket 1024d)
# ------------------------------------------------------------------------------
with tab_hybrid:
    st.subheader("Hybrid Search")
    st.caption("Searches your private index (768d) and an Invision bucket (1024d) in one call.")

    # Pre-fill from the Buckets tab selection if available
    default_bucket = st.session_state.get("selected_bucket", "enforcement-sanctions")
    default_index = st.session_state.get("selected_index", "sebi-orders")

    h_query = st.text_input(
        "Search query",
        value="SEBI front running compliance violations",
        key="h_query",
    )
    hc1, hc2 = st.columns(2)
    h_bucket = hc1.text_input("Invision bucket", value=default_bucket, key="h_bucket")
    h_index = hc2.text_input("Invision index", value=default_index, key="h_index")
    hc3, hc4 = st.columns(2)
    h_topk = hc3.slider("top_k", 1, 15, 5, key="h_topk")
    h_summarize = hc4.checkbox("Summarize", value=True, key="h_sum")

    if st.button("Run Hybrid Search", type="primary"):
        if not h_query.strip():
            st.warning("Please enter a query.")
        else:
            payload = {"query": h_query.strip(), "top_k": h_topk, "summarize": h_summarize}
            if h_bucket.strip():
                payload["invision_bucket"] = h_bucket.strip()
            if h_index.strip():
                payload["invision_index"] = h_index.strip()

            with st.spinner("Running hybrid search..."):
                try:
                    res, ms = timed_post(HYBRID_URL, api_key, payload, timeout=120)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            if res.status_code != 200:
                st.error(f"Hybrid search failed with HTTP {res.status_code} ({ms} ms): {res.text}")
            else:
                data = res.json()
                show_latency(ms, data.get("latency_ms"))

                # 1. Private index results (768d)
                private = data.get("private_index", {}) or {}
                st.markdown(f"### 🔒 Private Index `{private.get('assigned_index', '-')}`")
                if private.get("summary"):
                    st.info(f"**Summary:** {private['summary']}")
                p_results = private.get("results", []) or []
                if not p_results:
                    st.caption("No private index results.")
                for i, r in enumerate(p_results, 1):
                    meta = r.get("metadata", {}) or {}
                    with st.expander(
                        f"#{i} — Distance: {r.get('distance')} | Key: {r.get('key')}",
                        expanded=(i == 1),
                    ):
                        st.write(meta.get("text", "(No text)"))
                        if meta:
                            st.caption(f"Metadata: {meta}")

                # 2. Invision bucket results (1024d)
                invision = data.get("invision_bucket", {}) or {}
                st.markdown(
                    f"### 🪣 Invision Bucket `{invision.get('bucket', '-')}` / `{invision.get('index', '-')}`"
                )
                if invision.get("answer"):
                    st.success(f"**AI Answer:** {invision['answer']}")
                sources = invision.get("sources", []) or []
                if not sources:
                    st.caption("No Invision bucket sources.")
                for i, s in enumerate(sources, 1):
                    meta = s.get("metadata", {}) or {}
                    with st.expander(
                        f"#{i} — Distance: {s.get('distance')} | Key: {s.get('key')}",
                        expanded=(i == 1),
                    ):
                        st.write(meta.get("text", "(No text)"))
                        if meta:
                            st.caption(f"Metadata: {meta}")

                with st.expander("Raw JSON response"):
                    st.json(data)

# ------------------------------------------------------------------------------
# 4. BUCKETS (list_buckets action)
# ------------------------------------------------------------------------------
with tab_buckets:
    st.subheader("Invision Buckets")

    if st.button("List Buckets", type="primary"):
        with st.spinner("Fetching buckets..."):
            try:
                res, ms = timed_post(HYBRID_URL, api_key, {"action": "list_buckets"})
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if res.status_code == 200:
            data = res.json()
            st.session_state["buckets_raw"] = data
            st.session_state["buckets_map"] = extract_buckets(data)
            st.session_state["buckets_ms"] = ms
        else:
            st.error(f"list_buckets failed with HTTP {res.status_code} ({ms} ms): {res.text}")

    if "buckets_raw" in st.session_state:
        show_latency(st.session_state["buckets_ms"], st.session_state["buckets_raw"].get("latency_ms")
                     if isinstance(st.session_state["buckets_raw"], dict) else None)

        bmap = st.session_state.get("buckets_map", {})
        if bmap:
            bucket = st.selectbox("Bucket", list(bmap.keys()), key="bucket_pick")
            idx_options = bmap.get(bucket, [])
            if idx_options:
                index = st.selectbox("Index", idx_options, key="index_pick")
            else:
                index = st.text_input("Index (not listed — enter manually)", key="index_manual")

            if st.button("Use in Hybrid Search"):
                st.session_state["selected_bucket"] = bucket
                st.session_state["selected_index"] = index
                # Update widget values directly so the Hybrid tab reflects them
                st.session_state["h_bucket"] = bucket
                st.session_state["h_index"] = index
                st.success(f"Selected `{bucket}` / `{index}` — switch to the Hybrid Search tab.")
        else:
            st.caption("Couldn't auto-parse bucket names; see raw response below.")

        with st.expander("Raw JSON response"):
            st.json(st.session_state["buckets_raw"])
