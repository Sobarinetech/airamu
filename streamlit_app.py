import json
import time

import requests
import streamlit as st

# ------------------------------------------------------------------ config
DEFAULT_RAG_URL = "https://lbtoopahmulfgffzjumy.supabase.co/functions/v1/lens-rag-endpoint"
DEFAULT_SEARCH_URL = "https://lbtoopahmulfgffzjumy.supabase.co/functions/v1/vector-index-search"
MAX_CHARS = 19_900
MAX_ITEMS = 50

st.set_page_config(page_title="Lens RAG Tester", page_icon="🔎", layout="wide")

API_KEY = st.secrets.get("LENS_API_KEY")
RAG_URL = st.secrets.get("RAG_URL", DEFAULT_RAG_URL)
SEARCH_URL = st.secrets.get("SEARCH_URL", DEFAULT_SEARCH_URL)

if "log" not in st.session_state:
    st.session_state.log = []


# ------------------------------------------------------------------ helpers
def call_api(url: str, payload: dict, label: str, auto_retry: bool = False, status_box=None):
    """POST to the API. Returns (http_status, json_or_text). Optionally waits on 429."""
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    while True:
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
        except requests.RequestException as e:
            st.session_state.log.append(
                {"call": label, "status": "network error", "secs": round(time.time() - t0, 2), "note": str(e)[:80]}
            )
            return None, {"error": str(e)}

        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text}

        note = ""
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            note = f"Retry-After {wait}s"
        st.session_state.log.append(
            {"call": label, "status": r.status_code, "secs": round(time.time() - t0, 2), "note": note}
        )

        if r.status_code == 429 and auto_retry:
            for left in range(wait, 0, -1):
                if status_box:
                    status_box.warning(f"Rate limited (429). Retrying in {left}s…")
                time.sleep(1)
            continue
        return r.status_code, body


ERROR_HINTS = {
    400: "Bad input (missing field, >50 chunks, >19,900 chars, or bad vector).",
    401: "Missing / invalid / inactive / expired API key.",
    403: "No vector index assigned to this key — ask the Lens Boss admin.",
    405: "Endpoints accept POST only.",
    429: "Rate limit exceeded — wait for Retry-After seconds.",
    500: "Embedding / vector store error — safe to retry with backoff.",
}


def show_error(status, body):
    hint = ERROR_HINTS.get(status, "")
    st.error(f"HTTP {status}: {hint}")
    st.json(body)


def chunk_text(text: str, size: int, overlap: int):
    text = text.strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += max(size - overlap, 1)
    return out


def make_batches(chunks, max_items=MAX_ITEMS, max_chars=MAX_CHARS):
    cur, total = [], 0
    for c in chunks:
        n = len(c["text"])
        if cur and (len(cur) >= max_items or total + n > max_chars):
            yield cur
            cur, total = [], 0
        cur.append(c)
        total += n
    if cur:
        yield cur


def render_results(results):
    for i, res in enumerate(results, 1):
        meta = res.get("metadata") or {}
        title = meta.get("document_title") or meta.get("document_id") or ""
        page = meta.get("page_number")
        head = f"#{i} · distance {res.get('distance'):.4f} · {res.get('key')}"
        if title:
            head += f" · {title}"
        if page is not None:
            head += f" · p.{page}"
        with st.expander(head, expanded=i <= 3):
            st.write(meta.get("text", "_(no text in metadata)_"))
            st.caption("Metadata")
            st.json(meta)


# ------------------------------------------------------------------ guard
if not API_KEY:
    st.title("🔎 Lens RAG Tester")
    st.error("`LENS_API_KEY` not found in Streamlit secrets.")
    st.code('# .streamlit/secrets.toml\nLENS_API_KEY = "lens_xxxxxxxxxxxxxxxx"', language="toml")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Lens RAG Tester")
    st.caption(f"Key loaded: `{API_KEY[:9]}…{API_KEY[-4:]}`")
    st.markdown(
        "**Limits (per key)**\n"
        "- Embedding calls: 3/min, 100/3h\n"
        "- Searches: 100/3h\n"
        "- 19,900 chars per request\n"
        "- Upsert: 1–50 chunks"
    )
    st.divider()
    st.subheader("Session call log")
    if st.session_state.log:
        st.dataframe(st.session_state.log[::-1], use_container_width=True, hide_index=True)
        if st.button("Clear log"):
            st.session_state.log = []
            st.rerun()
    else:
        st.caption("No calls yet.")

tab_embed, tab_upsert, tab_query, tab_search = st.tabs(
    ["1 · Embed", "2 · Upsert", "3 · RAG query", "4 · Vector search"]
)

# ------------------------------------------------------------------ EMBED
with tab_embed:
    st.subheader("Embed text")
    st.caption("Returns a 768-dim vector. No index needed.")
    text = st.text_area("Text", "SEBI margin requirements for F&O", height=120, key="embed_text")
    st.caption(f"{len(text):,} / {MAX_CHARS:,} chars")
    if st.button("Embed", type="primary", key="btn_embed"):
        with st.spinner("Embedding…"):
            status, body = call_api(RAG_URL, {"mode": "embed", "inputs": text}, "embed")
        if status == 200 and body.get("success"):
            st.session_state.embed_result = {"input": text, **body}
        else:
            st.session_state.pop("embed_result", None)
            show_error(status, body)

    res = st.session_state.get("embed_result")
    if res:
        vec = res["embedding"]
        st.success(f"dimension = {res.get('dimension')} · returned {len(vec)} floats")
        st.write("First 8 values:", [round(v, 5) for v in vec[:8]])
        st.line_chart(vec, height=160)
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇ Download embedding (JSON object)",
            json.dumps(res, indent=2),
            "embedding.json",
            "application/json",
            help='{"input": ..., "success": true, "dimension": 768, "embedding": [...]}',
        )
        d2.download_button(
            "⬇ Download raw vector (array only)",
            json.dumps(vec),
            "embedding_vector.json",
            "application/json",
            help="Flat array of 768 floats — paste straight into the Vector search tab.",
        )

# ------------------------------------------------------------------ UPSERT
with tab_upsert:
    st.subheader("Upsert document chunks")
    c1, c2 = st.columns(2)
    doc_id = c1.text_input("document_id (required, ≤120)", "policy-credit-2026", max_chars=120)
    doc_title = c2.text_input("document_title (optional, ≤200)", "Credit Policy 2026", max_chars=200)

    source = st.radio("Chunk source", ["Paste text", "Upload PDF", "Raw JSON chunks"], horizontal=True)
    chunks = []

    if source in ("Paste text", "Upload PDF"):
        s1, s2 = st.columns(2)
        size = s1.slider("Chunk size (chars)", 200, 3000, 1000, 50)
        overlap = s2.slider("Overlap (chars)", 0, 500, 100, 10)

    if source == "Paste text":
        raw = st.text_area("Document text", height=220)
        for i, t in enumerate(chunk_text(raw, size, overlap) if raw.strip() else []):
            chunks.append({"chunk_id": str(i), "text": t})

    elif source == "Upload PDF":
        pdf = st.file_uploader("PDF", type="pdf")
        if pdf:
            from pypdf import PdfReader

            reader = PdfReader(pdf)
            n = 0
            for pno, page in enumerate(reader.pages, 1):
                ptxt = (page.extract_text() or "").strip()
                for t in chunk_text(ptxt, size, overlap) if ptxt else []:
                    chunks.append({"chunk_id": str(n), "text": t, "page_number": pno})
                    n += 1
            st.caption(f"{len(reader.pages)} pages → {len(chunks)} chunks")

    else:
        example = [{"chunk_id": "1", "text": "Maximum LTV for home loans is 80%.", "page_number": 1}]
        raw_json = st.text_area("JSON array of chunks", json.dumps(example, indent=2), height=220)
        try:
            parsed = json.loads(raw_json)
            chunks = [c for c in parsed if c.get("text")]
        except json.JSONDecodeError as e:
            st.warning(f"Invalid JSON: {e}")

    if chunks:
        batches = list(make_batches(chunks))
        total_chars = sum(len(c["text"]) for c in chunks)
        st.info(
            f"**{len(chunks)}** chunks · **{total_chars:,}** chars · **{len(batches)}** request(s). "
            f"Each request is one embedding call (limit 3/min) — "
            f"≈{max(len(batches) - 3, 0)} min of waiting if auto-retry is on."
        )
        with st.expander("Preview chunks"):
            st.dataframe(
                [{"chunk_id": c.get("chunk_id"), "page": c.get("page_number"), "chars": len(c["text"]), "text": c["text"][:120]} for c in chunks],
                use_container_width=True,
                hide_index=True,
            )

        # Chunks in the same shape the "Raw JSON chunks" input and the API expect
        export = []
        for c in chunks:
            item = {"chunk_id": str(c.get("chunk_id", "")), "text": c["text"]}
            if c.get("page_number") is not None:
                item["page_number"] = c["page_number"]
            export.append(item)
        st.download_button(
            "⬇ Download chunks (JSON)",
            json.dumps(export, indent=2, ensure_ascii=False),
            f"{(doc_id.strip() or 'chunks')}_chunks.json",
            "application/json",
        )

        auto_retry = st.checkbox("Auto-wait and retry on 429", value=True)
        if st.button("Upsert to index", type="primary", disabled=not doc_id.strip()):
            bar = st.progress(0.0)
            box = st.empty()
            done = 0
            ok = True
            for bi, batch in enumerate(batches, 1):
                payload = {"mode": "upsert", "document_id": doc_id.strip(), "chunks": batch}
                if doc_title.strip():
                    payload["document_title"] = doc_title.strip()
                box.info(f"Sending batch {bi}/{len(batches)} ({len(batch)} chunks)…")
                status, body = call_api(RAG_URL, payload, f"upsert {bi}/{len(batches)}", auto_retry, box)
                if status == 200 and body.get("success"):
                    done += body.get("upserted", len(batch))
                    bar.progress(bi / len(batches))
                else:
                    show_error(status, body)
                    ok = False
                    break
            if ok:
                box.success(f"Upserted {done} chunks into `{doc_id}`.")

# ------------------------------------------------------------------ QUERY
with tab_query:
    st.subheader("RAG query")
    st.caption("Embeds your question and returns the closest chunks. Uses embedding **and** search quota.")
    q = st.text_input("Question", "What is the maximum LTV?", key="q_query")
    k = st.slider("top_k", 1, 20, 5, key="k_query")
    if st.button("Run query", type="primary", key="btn_query"):
        with st.spinner("Querying…"):
            status, body = call_api(RAG_URL, {"mode": "query", "inputs": q, "top_k": k}, "query")
        if status == 200:
            results = body.get("results") or body.get("matches") or []
            st.success(f"{len(results)} result(s)")
            render_results(results)
            with st.expander("Raw response"):
                st.json(body)
        else:
            show_error(status, body)

# ------------------------------------------------------------------ SEARCH
with tab_search:
    st.subheader("Vector index search")
    st.caption("Search with text (we embed it) or your own 768-dim vector. Lower distance = closer.")
    mode = st.radio("Input", ["Text", "Custom vector (768 floats)"], horizontal=True)
    k2 = st.slider("top_k", 1, 30, 5, key="k_search")
    payload = None

    if mode == "Text":
        qs = st.text_input("Query", "margin call trigger", key="q_search")
        if st.button("Search", type="primary", key="btn_search_text"):
            payload = {"query": qs, "top_k": k2}
    else:
        vtxt = st.text_area("JSON array of 768 numbers", height=140, key="vec_in")
        if st.button("Search", type="primary", key="btn_search_vec"):
            try:
                vec = json.loads(vtxt)
                if not (isinstance(vec, list) and len(vec) == 768):
                    st.error(f"Need exactly 768 numbers, got {len(vec) if isinstance(vec, list) else 'non-list'}.")
                else:
                    payload = {"queryVector": vec, "top_k": k2}
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    if payload:
        with st.spinner("Searching…"):
            status, body = call_api(SEARCH_URL, payload, "vector-index-search")
        if status == 200:
            st.success(f"{body.get('count', 0)} result(s)")
            render_results(body.get("results", []))
            with st.expander("Raw response"):
                st.json(body)
        else:
            show_error(status, body)
