"""
IBBI BaankNet Auction Scraper — Browserless Test Console
=========================================================

A Streamlit app to test scraping https://ibbi.baanknet.com/eauction-ibbi/home
using a self-hosted Browserless REST API instance, then optionally push
discovered documents to a PDF-extraction endpoint.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import re
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------

DEFAULT_BROWSERLESS_URL = "http://72.61.251.247:32768"
DEFAULT_PDF_EXTRACT_URL = "http://72.61.251.247:3000/extract-pdf-url"
IBBI_BASE = "https://ibbi.baanknet.com"
IBBI_HOME = f"{IBBI_BASE}/eauction-ibbi/home"

st.set_page_config(page_title="IBBI BaankNet Scraper Console", layout="wide")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "listings" not in st.session_state:
    st.session_state.listings = []
if "last_raw_html" not in st.session_state:
    st.session_state.last_raw_html = ""
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = []

# --------------------------------------------------------------------------
# Sidebar — connection settings
# --------------------------------------------------------------------------

st.sidebar.header("Connection settings")
browserless_url = st.sidebar.text_input(
    "Browserless base URL", value=DEFAULT_BROWSERLESS_URL,
    help="e.g. http://72.61.251.247:32768"
).rstrip("/")
browserless_token = st.sidebar.text_input(
    "Browserless API token (optional)", value="", type="password",
    help="Leave blank if your self-hosted instance doesn't require one."
)
request_timeout = st.sidebar.number_input(
    "Request timeout (seconds)", min_value=5, max_value=300, value=60, step=5
)

st.sidebar.divider()
pdf_extract_url = st.sidebar.text_input(
    "PDF extraction endpoint", value=DEFAULT_PDF_EXTRACT_URL
)

st.sidebar.divider()
st.sidebar.caption(
    "This tool sends requests from wherever Streamlit is running to your "
    "Browserless instance. Make sure that host/port is reachable from here."
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _endpoint(path: str) -> str:
    url = f"{browserless_url}/{path.lstrip('/')}"
    if browserless_token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={browserless_token}"
    return url


def call_browserless(path: str, payload: dict):
    """POST to a Browserless REST endpoint. Returns (ok, status_code, response_obj_or_text, content_type)."""
    url = _endpoint(path)
    try:
        resp = requests.post(url, json=payload, timeout=request_timeout)
    except requests.exceptions.RequestException as e:
        return False, None, str(e), None

    content_type = resp.headers.get("content-type", "")
    ok = resp.ok
    if "application/json" in content_type:
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
    else:
        body = resp.text
    return ok, resp.status_code, body, content_type


def fetch_rendered_html(url: str, wait_selector: str | None = None):
    """Use /content to get fully rendered HTML for a given URL."""
    payload = {"url": url, "gotoOptions": {"waitUntil": "networkidle2"}}
    if wait_selector:
        payload["waitForSelector"] = {"selector": wait_selector, "timeout": 15000}
    ok, status, body, ctype = call_browserless("content", payload)
    if not ok:
        return None, f"HTTP {status}: {body}"
    if isinstance(body, dict):
        # some browserless configs wrap html in JSON
        html = body.get("data") or body.get("html") or json.dumps(body)
    else:
        html = body
    return html, None


def text_or_none(tag):
    return tag.get_text(strip=True) if tag else None


def parse_listing_block(block, page_url: str) -> dict:
    """Parse one .asset-listing-content block into a flat dict."""
    data = {}

    title_tag = block.select_one(".details-title h4")
    data["title"] = text_or_none(title_tag)

    summary_tag = block.select_one(".summary")
    data["summary"] = text_or_none(summary_tag)

    # label/value rows like "Asset ID :" -> "4375"
    for row in block.select(".items .row"):
        label_div = row.select_one(".data")
        value_span = row.select_one("span")
        if label_div and value_span:
            key = label_div.get_text(strip=True).rstrip(":").strip()
            data[key] = value_span.get_text(strip=True)

    company_tag = block.select_one(".company-title h4")
    data["company_name"] = text_or_none(company_tag)

    cin_tag = block.select_one(".company small")
    data["cin"] = text_or_none(cin_tag)

    img_tag = block.select_one(".img img")
    if img_tag and img_tag.get("src"):
        data["image_url"] = img_tag["src"]

    detail_link = block.select_one('.contact-btn a[title="View Asset Detail"]')
    if detail_link and detail_link.get("href"):
        data["asset_detail_url"] = urljoin(page_url, detail_link["href"])

    status_tag = block.select_one(".open-tag span")
    data["status"] = text_or_none(status_tag)

    return data


def parse_listings(html: str, page_url: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.select(".eproc-listing-main .asset-listing-content")
    return [parse_listing_block(b, page_url) for b in blocks]


def extract_document_links(html: str, page_url: str) -> list:
    """Find 'Download Document' style links on an asset detail page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "view-home-company-asset-doc" in href or "download" in a.get_text(strip=True).lower():
            links.append({
                "text": a.get_text(strip=True),
                "url": urljoin(page_url, href),
            })
    return links


def call_pdf_extractor(pdf_url: str):
    try:
        resp = requests.post(
            pdf_extract_url,
            json={"url": pdf_url},
            headers={"Content-Type": "application/json"},
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as e:
        return False, str(e)
    if not resp.ok:
        return False, f"HTTP {resp.status_code}: {resp.text}"
    try:
        return True, resp.json()
    except ValueError:
        return True, resp.text


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🏛️ IBBI BaankNet Auction Scraper — Browserless Console")
st.caption(
    "Test scraping of ibbi.baanknet.com eAuctions through your self-hosted "
    "Browserless REST API, and pipe discovered documents to your PDF extractor."
)

tab_raw, tab_listings, tab_detail, tab_pdf, tab_pipeline = st.tabs(
    ["🔧 Raw request tester", "📋 Auction listings", "📄 Asset detail", "🧾 PDF extractor", "🚀 Full pipeline"]
)

# ---- Tab 1: Raw request tester -------------------------------------------
with tab_raw:
    st.subheader("Hit any Browserless REST endpoint")
    col1, col2 = st.columns([1, 2])
    with col1:
        endpoint_choice = st.selectbox(
            "Endpoint",
            ["content", "scrape", "smart-scrape", "screenshot", "function"],
            index=0,
        )
    with col2:
        target_url = st.text_input("Target URL", value=IBBI_HOME)

    default_payloads = {
        "content": {"url": target_url, "gotoOptions": {"waitUntil": "networkidle2"}},
        "scrape": {
            "url": target_url,
            "elements": [{"selector": ".asset-listing-content"}],
        },
        "smart-scrape": {"url": target_url, "format": ["html", "markdown"]},
        "screenshot": {"url": target_url, "options": {"fullPage": True}},
        "function": {
            "code": (
                "export default async ({ page }) => {\n"
                "  await page.goto(context.url, { waitUntil: 'networkidle2' });\n"
                "  const html = await page.content();\n"
                "  return { data: html, type: 'text/html' };\n"
                "};"
            ),
            "context": {"url": target_url},
        },
    }

    payload_text = st.text_area(
        "Request payload (JSON)",
        value=json.dumps(default_payloads[endpoint_choice], indent=2),
        height=220,
    )

    if st.button("Send request", type="primary"):
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON payload: {e}")
        else:
            with st.spinner(f"Calling /{endpoint_choice} ..."):
                ok, status, body, ctype = call_browserless(endpoint_choice, payload)
            st.write(f"**Status:** {status}  |  **Content-Type:** {ctype}")
            if not ok:
                st.error("Request failed")
            if isinstance(body, (dict, list)):
                st.json(body)
            else:
                st.session_state.last_raw_html = body if isinstance(body, str) else ""
                with st.expander("Raw response", expanded=True):
                    st.code(body[:20000] if isinstance(body, str) else str(body), language="html")

# ---- Tab 2: Auction listings ----------------------------------------------
with tab_listings:
    st.subheader("Fetch & parse the auction home page")
    home_url = st.text_input("Home page URL", value=IBBI_HOME, key="home_url")

    if st.button("Fetch listings", type="primary"):
        with st.spinner("Rendering page via Browserless /content ..."):
            html, err = fetch_rendered_html(home_url, wait_selector=".asset-listing-content")
        if err:
            st.error(err)
        else:
            st.session_state.last_raw_html = html
            listings = parse_listings(html, home_url)
            st.session_state.listings = listings
            st.success(f"Parsed {len(listings)} listing(s).")

    if st.session_state.listings:
        df = pd.DataFrame(st.session_state.listings)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download listings as JSON",
            data=json.dumps(st.session_state.listings, indent=2),
            file_name="ibbi_listings.json",
            mime="application/json",
        )
        st.download_button(
            "Download listings as CSV",
            data=df.to_csv(index=False),
            file_name="ibbi_listings.csv",
            mime="text/csv",
        )

    with st.expander("View last raw HTML fetched"):
        st.code(
            (st.session_state.last_raw_html or "")[:20000],
            language="html",
        )

# ---- Tab 3: Asset detail ---------------------------------------------------
with tab_detail:
    st.subheader("Fetch a single asset detail page & find its document link")

    options = ["(enter manually)"] + [
        f"{l.get('Asset ID', '?')} — {l.get('title', '')}" for l in st.session_state.listings
    ]
    choice = st.selectbox("Pick from parsed listings, or enter manually", options)

    if choice != "(enter manually)":
        idx = options.index(choice) - 1
        default_detail_url = st.session_state.listings[idx].get("asset_detail_url", "")
    else:
        default_detail_url = ""

    detail_url = st.text_input("Asset detail URL", value=default_detail_url)

    if st.button("Fetch asset detail page", type="primary"):
        if not detail_url:
            st.warning("Enter or select an asset detail URL first.")
        else:
            with st.spinner("Rendering asset detail page ..."):
                html, err = fetch_rendered_html(detail_url)
            if err:
                st.error(err)
            else:
                st.session_state.last_raw_html = html
                docs = extract_document_links(html, detail_url)
                if docs:
                    st.success(f"Found {len(docs)} document link(s).")
                    for d in docs:
                        st.write(f"- **{d['text'] or 'Document'}**: {d['url']}")
                    st.session_state["detail_docs"] = docs
                else:
                    st.warning("No document links found on this page.")
                with st.expander("View rendered HTML"):
                    st.code(html[:20000], language="html")

# ---- Tab 4: PDF extractor --------------------------------------------------
with tab_pdf:
    st.subheader("Send a document URL to your PDF extraction endpoint")

    doc_options = ["(enter manually)"] + [
        d["url"] for d in st.session_state.get("detail_docs", [])
    ]
    doc_choice = st.selectbox("Pick a found document, or enter manually", doc_options)
    default_doc_url = "" if doc_choice == "(enter manually)" else doc_choice

    doc_url = st.text_input("Document URL", value=default_doc_url)

    if st.button("Extract PDF text", type="primary"):
        if not doc_url:
            st.warning("Enter a document URL first.")
        else:
            with st.spinner("Calling PDF extraction endpoint ..."):
                ok, result = call_pdf_extractor(doc_url)
            if not ok:
                st.error(result)
            else:
                st.success("Extraction complete.")
                if isinstance(result, dict):
                    st.json(result)
                    text_val = result.get("text") or result.get("data") or ""
                else:
                    text_val = str(result)
                    st.text_area("Extracted text", value=text_val, height=400)
                if text_val:
                    st.download_button(
                        "Download extracted text",
                        data=text_val,
                        file_name="extracted.txt",
                        mime="text/plain",
                    )

# ---- Tab 5: Full pipeline --------------------------------------------------
with tab_pipeline:
    st.subheader("Run the full pipeline: listings → detail page → document → extracted text")
    st.caption(
        "Fetches the home page, parses each listing, opens its asset-detail page, "
        "finds the document link, and runs it through the PDF extractor."
    )

    max_items = st.number_input(
        "Max listings to process (0 = all currently parsed listings)",
        min_value=0, value=3, step=1,
    )

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        refresh_first = st.checkbox("Re-fetch listings first", value=not bool(st.session_state.listings))
    with run_col2:
        pipeline_home_url = st.text_input("Home URL for pipeline", value=IBBI_HOME, key="pipeline_home_url")

    if st.button("▶ Run pipeline", type="primary"):
        if refresh_first or not st.session_state.listings:
            with st.spinner("Fetching & parsing home page ..."):
                html, err = fetch_rendered_html(pipeline_home_url, wait_selector=".asset-listing-content")
            if err:
                st.error(err)
                st.stop()
            st.session_state.listings = parse_listings(html, pipeline_home_url)

        listings = st.session_state.listings
        if max_items and max_items > 0:
            listings = listings[: int(max_items)]

        results = []
        progress = st.progress(0.0, text="Starting pipeline ...")
        total = max(len(listings), 1)

        for i, listing in enumerate(listings):
            row = {
                "asset_id": listing.get("Asset ID"),
                "title": listing.get("title"),
                "asset_detail_url": listing.get("asset_detail_url"),
                "document_url": None,
                "extracted_text": None,
                "error": None,
            }
            detail_url = listing.get("asset_detail_url")
            progress.progress((i) / total, text=f"Processing {row['title'] or row['asset_id']} ...")

            if not detail_url:
                row["error"] = "No asset_detail_url found on listing"
                results.append(row)
                continue

            html, err = fetch_rendered_html(detail_url)
            if err:
                row["error"] = f"detail fetch failed: {err}"
                results.append(row)
                continue

            docs = extract_document_links(html, detail_url)
            if not docs:
                row["error"] = "no document link found"
                results.append(row)
                continue

            row["document_url"] = docs[0]["url"]
            ok, result = call_pdf_extractor(docs[0]["url"])
            if not ok:
                row["error"] = f"pdf extraction failed: {result}"
            else:
                if isinstance(result, dict):
                    row["extracted_text"] = result.get("text") or result.get("data") or json.dumps(result)
                else:
                    row["extracted_text"] = str(result)

            results.append(row)
            progress.progress((i + 1) / total, text=f"Done {row['title'] or row['asset_id']}")

        st.session_state.pipeline_results = results
        progress.empty()
        st.success(f"Pipeline finished — processed {len(results)} listing(s).")

    if st.session_state.pipeline_results:
        df = pd.DataFrame(st.session_state.pipeline_results)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download pipeline results as JSON",
            data=json.dumps(st.session_state.pipeline_results, indent=2),
            file_name="ibbi_pipeline_results.json",
            mime="application/json",
        )
