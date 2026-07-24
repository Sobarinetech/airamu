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
    "Request timeout (seconds)", min_value=5, max_value=300, value=90, step=5
)

fetch_method = st.sidebar.selectbox(
    "Page-fetch method",
    ["content (spoofed headers)", "function (stealth script)"],
    index=1,
    help=(
        "This self-hosted Browserless build does NOT register /unblock, "
        "/smart-scrape, /search, /map, or /crawl (those are cloud/paid-tier "
        "routes) — only /content, /scrape, /screenshot, /pdf, /function, "
        "/download are available. ibbi.baanknet.com 403s the very first "
        "request (even favicon.ico), which looks like an IP/WAF-level block "
        "on the VPS's datacenter IP rather than pure JS fingerprinting. "
        "'/function' lets us spoof UA/headers and strip navigator.webdriver "
        "before navigating, which is the best shot available on this build."
    ),
)

custom_user_agent = st.sidebar.text_input(
    "Spoofed User-Agent",
    value=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
)
custom_accept_language = st.sidebar.text_input(
    "Accept-Language header", value="en-IN,en-US;q=0.9,en;q=0.8"
)

proxy_server = st.sidebar.text_input(
    "Proxy server (optional)",
    value="",
    placeholder="http://user:pass@host:port",
    help=(
        "ibbi.baanknet.com appears to block this VPS's IP outright — no "
        "amount of header/UA spoofing fixes an IP-level block. If you have "
        "a residential/rotating proxy, set it here and Browserless will "
        "launch Chromium through it. Leave blank to keep using the VPS's "
        "direct connection (will likely stay blocked — use Diagnostics to "
        "confirm)."
    ),
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


def _looks_blocked(html: str) -> bool:
    """Heuristic: does this look like a WAF/error page rather than real content?"""
    if not html or len(html.strip()) < 500:
        return True
    lowered = html.lower()
    markers = ["403 forbidden", "access denied", "request blocked", "attention required"]
    return any(m in lowered for m in markers) and "asset-listing-content" not in lowered


def _launch_options():
    """Shared launchOptions payload fragment — adds a proxy server if configured."""
    if not proxy_server:
        return {}
    # Strip credentials for --proxy-server (Chromium doesn't accept user:pass in
    # the flag itself); auth is handled separately per-endpoint where possible.
    from urllib.parse import urlparse
    parsed = urlparse(proxy_server if "://" in proxy_server else f"http://{proxy_server}")
    server = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}" if parsed.hostname else proxy_server
    return {"launchOptions": {"args": [f"--proxy-server={server}"]}}


def fetch_via_content(url: str, wait_selector: str | None):
    """Use /content — spoofs a normal browser UA/headers. May still be IP-blocked."""
    payload = {
        "url": url,
        "gotoOptions": {"waitUntil": "networkidle2", "timeout": 45000},
        "setJavaScriptEnabled": True,
        "userAgent": custom_user_agent,
        "setExtraHTTPHeaders": {
            "Accept-Language": custom_accept_language,
            "Upgrade-Insecure-Requests": "1",
        },
        # bestAttempt: still return whatever HTML we got even if the
        # waitForSelector below times out, instead of a hard 500.
        "bestAttempt": True,
        **_launch_options(),
    }
    if wait_selector:
        payload["waitForSelector"] = {"selector": wait_selector, "timeout": 20000}
    ok, status, body, ctype = call_browserless("content", payload)
    if not ok:
        return None, f"/content failed — HTTP {status}: {body}"
    html = body.get("data") if isinstance(body, dict) else body
    if html is None:
        html = json.dumps(body)
    return html, None


_FUNCTION_STEALTH_TEMPLATE = r"""
export default async ({ page, context }) => {
  const { url, userAgent, acceptLanguage, waitSelector, proxyUsername, proxyPassword } = context;

  if (proxyUsername) {
    await page.authenticate({ username: proxyUsername, password: proxyPassword || '' });
  }

  // Basic fingerprint cleanup before anything loads
  await page.evaluateOnNewDocument(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
    window.chrome = { runtime: {} };
  });

  await page.setUserAgent(userAgent);
  await page.setExtraHTTPHeaders({
    'Accept-Language': acceptLanguage,
    'Upgrade-Insecure-Requests': '1',
  });
  await page.setViewport({ width: 1366, height: 900 });

  let status = null;
  try {
    const resp = await page.goto(url, { waitUntil: 'networkidle2', timeout: 45000 });
    status = resp ? resp.status() : null;
  } catch (e) {
    // fall through — we still try to grab whatever content is there
  }

  if (waitSelector) {
    try {
      await page.waitForSelector(waitSelector, { timeout: 20000 });
    } catch (e) {
      // ignore — bestAttempt behavior, return what we have
    }
  }

  const html = await page.content();
  return { data: JSON.stringify({ html, status }), type: 'application/json' };
};
"""


def fetch_via_function(url: str, wait_selector: str | None):
    """Use /function — the only route on this build that lets us spoof UA/headers
    and strip navigator.webdriver before navigating."""
    proxy_user, proxy_pass = None, None
    if proxy_server and "@" in proxy_server:
        from urllib.parse import urlparse
        p = urlparse(proxy_server if "://" in proxy_server else f"http://{proxy_server}")
        proxy_user, proxy_pass = p.username, p.password

    payload = {
        "code": _FUNCTION_STEALTH_TEMPLATE,
        "context": {
            "url": url,
            "userAgent": custom_user_agent,
            "acceptLanguage": custom_accept_language,
            "waitSelector": wait_selector or None,
            "proxyUsername": proxy_user,
            "proxyPassword": proxy_pass,
        },
        **_launch_options(),
    }
    ok, status, body, ctype = call_browserless("function", payload)
    if not ok:
        return None, f"/function failed — HTTP {status}: {body}"

    # body may already be parsed JSON, or a raw string containing JSON
    parsed = body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except ValueError:
            return body, None  # not JSON — treat as raw html fallback

    if isinstance(parsed, dict) and "html" in parsed:
        html = parsed["html"]
        page_status = parsed.get("status")
        if page_status and page_status >= 400:
            return html, f"Target page responded with HTTP {page_status} inside the browser."
        return html, None

    return json.dumps(parsed), None


def fetch_rendered_html(url: str, wait_selector: str | None = None, method: str | None = None):
    """Fetch fully rendered HTML for a URL, routing through the selected method.

    Falls back gracefully: if the primary method returns a blocked/empty page,
    tries the other method once before giving up.
    """
    method = method or fetch_method
    primary = fetch_via_function if method.startswith("function") else fetch_via_content
    fallback = fetch_via_content if method.startswith("function") else fetch_via_function

    html, err = primary(url, wait_selector)
    if err:
        html2, err2 = fallback(url, wait_selector)
        if err2:
            return None, f"Primary method failed ({err}); fallback also failed ({err2})"
        if _looks_blocked(html2):
            return html2, f"Both methods returned what looks like a blocked/empty page. Last error: {err}"
        return html2, None

    if _looks_blocked(html):
        html2, err2 = fallback(url, wait_selector)
        if not err2 and html2 and not _looks_blocked(html2):
            return html2, None
        return html, ("Warning: response looks like a WAF/blocked page (403 or near-empty). "
                       "Try the other fetch method in the sidebar, or increase the timeout.")

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

tab_diag, tab_raw, tab_listings, tab_detail, tab_pdf, tab_pipeline = st.tabs(
    ["🩺 Diagnostics", "🔧 Raw request tester", "📋 Auction listings", "📄 Asset detail", "🧾 PDF extractor", "🚀 Full pipeline"]
)

# ---- Tab 0: Diagnostics ----------------------------------------------------
with tab_diag:
    st.subheader("Is this an IP block or a fingerprint block?")
    st.caption(
        "ibbi.baanknet.com 403s the very first request from this Browserless "
        "instance — even favicon.ico, before any JS runs. That pattern points "
        "to an IP/WAF-level block on the VPS's egress IP. These two checks "
        "confirm it."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**1. What IP does Browserless actually egress from?**")
        if st.button("Check Browserless's outbound IP"):
            ip_check_code = (
                "export default async ({ page, context }) => {\n"
                "  if (context.proxyUsername) {\n"
                "    await page.authenticate({ username: context.proxyUsername, password: context.proxyPassword || '' });\n"
                "  }\n"
                "  const resp = await page.goto('https://api.ipify.org?format=json', { waitUntil: 'networkidle2', timeout: 20000 });\n"
                "  const text = await resp.text();\n"
                "  return { data: text, type: 'application/json' };\n"
                "};"
            )
            proxy_user, proxy_pass = None, None
            if proxy_server and "@" in proxy_server:
                from urllib.parse import urlparse
                p = urlparse(proxy_server if "://" in proxy_server else f"http://{proxy_server}")
                proxy_user, proxy_pass = p.username, p.password
            payload = {
                "code": ip_check_code,
                "context": {"proxyUsername": proxy_user, "proxyPassword": proxy_pass},
                **_launch_options(),
            }
            with st.spinner("Asking Browserless to report its outbound IP ..."):
                ok, status, body, ctype = call_browserless("function", payload)
            if ok:
                st.success("Browserless is egressing from:")
                st.json(body if isinstance(body, dict) else json.loads(body))
                st.caption(
                    "Compare this to your VPS's known public IP "
                    f"(from the sidebar URL: {browserless_url}). If they match "
                    "and no proxy is set, that IP is what IBBI's WAF sees."
                )
            else:
                st.error(f"HTTP {status}: {body}")

    with col_b:
        st.markdown("**2. Plain HTTP request (no browser) from wherever this app runs**")
        diag_url = st.text_input("URL to test", value=IBBI_HOME, key="diag_url")
        if st.button("Send plain requests.get()"):
            try:
                headers = {
                    "User-Agent": custom_user_agent,
                    "Accept-Language": custom_accept_language,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                resp = requests.get(diag_url, headers=headers, timeout=request_timeout, allow_redirects=True)
                st.write(f"**Status:** {resp.status_code}")
                st.write(f"**Server header:** {resp.headers.get('server', 'n/a')}")
                with st.expander("Response headers"):
                    st.json(dict(resp.headers))
                with st.expander("First 2000 chars of body"):
                    st.code(resp.text[:2000])
            except requests.exceptions.RequestException as e:
                st.error(str(e))
        st.caption(
            "This request comes from wherever Streamlit itself is hosted "
            "(not the Browserless VPS), so a different result here vs. the "
            "Browserless calls tells you the block is specific to the VPS IP."
        )

    st.divider()
    st.markdown(
        "**If both checks confirm an IP-level block:** header/UA/fingerprint "
        "spoofing won't help — you'd need to route Browserless traffic "
        "through a residential/rotating proxy (set one in the sidebar) or "
        "run Browserless from a non-datacenter network."
    )

# ---- Tab 1: Raw request tester -------------------------------------------
with tab_raw:
    st.subheader("Hit any Browserless REST endpoint")
    col1, col2 = st.columns([1, 2])
    with col1:
        endpoint_choice = st.selectbox(
            "Endpoint",
            ["function", "content", "scrape", "screenshot", "pdf", "download"],
            index=0,
            help=(
                "Only these routes are registered on your self-hosted instance "
                "(per its boot log) — no /unblock, /smart-scrape, /search, "
                "/map, or /crawl. ibbi.baanknet.com 403s plain /content — "
                "/function (with UA/header spoofing) is the best shot here."
            ),
        )
    with col2:
        target_url = st.text_input("Target URL", value=IBBI_HOME)

    default_payloads = {
        "content": {
            "url": target_url,
            "gotoOptions": {"waitUntil": "networkidle2", "timeout": 45000},
            "userAgent": custom_user_agent,
            "setExtraHTTPHeaders": {"Accept-Language": custom_accept_language},
            "bestAttempt": True,
        },
        "scrape": {
            "url": target_url,
            "elements": [{"selector": ".asset-listing-content"}],
        },
        "screenshot": {"url": target_url, "options": {"fullPage": True}},
        "pdf": {"url": target_url},
        "download": {
            "code": (
                "export default async ({ page }) => {\n"
                "  await page.goto(context.url, { waitUntil: 'networkidle2' });\n"
                "};"
            ),
            "context": {"url": target_url},
        },
        "function": {
            "code": _FUNCTION_STEALTH_TEMPLATE.strip(),
            "context": {
                "url": target_url,
                "userAgent": custom_user_agent,
                "acceptLanguage": custom_accept_language,
                "waitSelector": ".asset-listing-content",
            },
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
        with st.spinner(f"Rendering page via Browserless ({fetch_method.split()[0]}) ..."):
            html, err = fetch_rendered_html(home_url, wait_selector=".asset-listing-content")
        if html is None:
            st.error(err)
        else:
            if err:
                st.warning(err)
            st.session_state.last_raw_html = html
            listings = parse_listings(html, home_url)
            st.session_state.listings = listings
            if listings:
                st.success(f"Parsed {len(listings)} listing(s).")
            else:
                st.warning(
                    "0 listings parsed. The page HTML was fetched but no "
                    "`.asset-listing-content` blocks were found — check the "
                    "raw HTML below to see what actually came back."
                )

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
            if html is None:
                st.error(err)
            else:
                if err:
                    st.warning(err)
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
            if html is None:
                st.error(err)
                st.stop()
            if err:
                st.warning(err)
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
            if html is None:
                row["error"] = f"detail fetch failed: {err}"
                results.append(row)
                continue
            if err:
                row["error"] = f"warning: {err}"

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
