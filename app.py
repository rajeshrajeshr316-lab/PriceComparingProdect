import csv
import io
import json
import random
import time
from urllib.parse import quote_plus

import requests
import streamlit as st
from google import genai
from google.genai import types

try:
    from PIL import Image   # only used to shrink big photos
except Exception:
    Image = None

# The app tries these models in order. Edit the list if a name stops working.
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]

st.set_page_config(
    page_title="PriceLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background-color: black; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 PriceLens")
st.caption("Snap a product. Find the best price.")


# ---------------------------------------------------------------
# Keys: read from .streamlit/secrets.toml, else ask in the sidebar
# ---------------------------------------------------------------
def get_key(name, label):
    try:
        return st.secrets[name]
    except Exception:
        with st.sidebar:
            return st.text_input(label, type="password")


gemini_key = get_key("GEMINI_API_KEY", "Gemini API key")
serp_key = get_key("SERPAPI_KEY", "SerpAPI key")

with st.sidebar:
    force_demo = st.checkbox("Use demo prices (no SerpAPI search)")
    st.markdown("---")
    st.caption("PriceLens: AI product recognition + price comparison")


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def friendly_error(e):
    """Turn a technical error into a short message a user can understand."""
    msg = str(e)
    if "API key" in msg or "API_KEY_INVALID" in msg:
        return "The API key looks invalid. Copy it again without extra spaces."
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "Free quota reached. Wait a minute, or try another model name."
    if "503" in msg or "UNAVAILABLE" in msg:
        return "Google's AI is busy right now. Wait a minute and click Identify again."
    if "404" in msg or "NOT_FOUND" in msg:
        return "Model not found. Edit the GEMINI_MODELS list at the top of app.py."
    if "Connection" in msg or "timed out" in msg:
        return "Network problem. Check your internet or try a mobile hotspot."
    return f"Something went wrong: {msg}"


def prepare_image(uploaded_file):
    """Shrink big photos to JPEG (if Pillow works). Returns (bytes, mime type)."""
    raw = uploaded_file.getvalue()
    if Image is None:
        return raw, uploaded_file.type
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        return raw, uploaded_file.type


def rows_to_csv(rows):
    """Make CSV text with Python's csv module (no pandas)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["Store", "Product", "Price (₹)", "Rating", "Link"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def store_search_url(store, query):
    """Build a search link for a retailer used by the demo results."""
    search_urls = {
        "Amazon.in": "https://www.amazon.in/s?k=",
        "Flipkart": "https://www.flipkart.com/search?q=",
        "Croma": "https://www.croma.com/searchB?q=",
        "Reliance Digital": "https://www.reliancedigital.in/search?q=",
        "Snapdeal": "https://www.snapdeal.com/search?keyword=",
        "Meesho": "https://www.meesho.com/search?q=",
    }
    base_url = search_urls.get(store, "https://www.google.com/search?q=")
    return base_url + quote_plus(query)


def identify_product(image_bytes, mime_type, key):
    client = genai.Client(api_key=key)
    prompt = (
        "Identify the main product in this image. Reply ONLY with JSON: "
        '{"name": "", "brand": "", "category": "", "search_query": ""}. '
        "search_query should be what a shopper in India would type to buy it."
    )
    last_error = None

    for model in GEMINI_MODELS:              # try each model in turn
        for attempt in range(2):             # try each model up to 2 times
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                return json.loads(response.text)
            except Exception as e:
                last_error = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2)            # Google is busy: wait, then retry / next model
                    continue
                if "404" in msg or "NOT_FOUND" in msg or "429" in msg:
                    break                    # this model is not usable: go to the next one
                raise                        # any other error: stop and show it

    raise last_error


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices_serpapi(query, key):
    """Live prices. Cached for 1 hour so the same search doesn't use up free quota."""
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google_shopping",
            "q": query,
            "gl": "in",
            "hl": "en",
            "api_key": key,
        },
        timeout=30,
    )
    response.raise_for_status()
    rows = []
    for item in response.json().get("shopping_results", []):
        price = item.get("extracted_price")
        if price:
            rows.append(
                {
                    "Store": item.get("source") or "Unknown",
                    "Product": item.get("title"),
                    "Price (₹)": price,
                    "Rating": item.get("rating"),
                    "Link": item.get("product_link") or item.get("link"),
                }
            )
    return rows


def demo_prices(query):
    """Fake but stable prices, used for testing or as a backup."""
    random.seed(query)
    base = random.randint(500, 5000)
    stores = ["Amazon.in", "Flipkart", "Croma", "Reliance Digital", "Snapdeal", "Meesho"]
    rows = []
    for store in stores:
        rows.append(
            {
                "Store": store,
                "Product": query,
                "Price (₹)": int(base * random.uniform(0.85, 1.15)),
                "Rating": round(random.uniform(3.8, 4.7), 1),
                "Link": store_search_url(store, query),
            }
        )
    return rows


# ---------------------------------------------------------------
# Main screen
# ---------------------------------------------------------------
file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])

if not file:
    st.info("👆 Upload a product photo to get started.")
    st.stop()

# A new image was uploaded -> clear the old results
if st.session_state.get("file_name") != file.name:
    st.session_state.file_name = file.name
    st.session_state.info = None
    st.session_state.rows = None
    st.session_state.is_demo = False

try:
    image_bytes, mime_type = prepare_image(file)
except Exception:
    st.error("Could not read this image. Please try another file.")
    st.stop()

col_left, col_right = st.columns([1, 2])

# ----- Left column: image + identify button -----
with col_left:
    st.image(file, width=300)

    if st.button("🔍 Identify product"):
        if not gemini_key:
            st.error("Please add your Gemini API key.")
        else:
            with st.spinner("AI is looking at your image..."):
                try:
                    st.session_state.info = identify_product(image_bytes, mime_type, gemini_key)
                    st.session_state.rows = None
                except Exception as e:
                    st.error(friendly_error(e))

# ----- Right column: product info + prices -----
with col_right:
    info = st.session_state.get("info")

    if not info:
        st.write("Click **Identify product** to let the AI recognise the item.")
    else:
        with st.chat_message("assistant"):
            st.write(
                f"I think this is **{info.get('name')}** "
                f"({info.get('brand')}, {info.get('category')})."
            )

        query = st.text_input(
            "Not correct? Edit the search text:", value=info.get("search_query", "")
        )

        if st.button("💰 Find best prices"):
            with st.spinner("Searching stores..."):
                rows, is_demo = [], False

                if serp_key and not force_demo:
                    try:
                        rows = get_prices_serpapi(query, serp_key)
                    except Exception as e:
                        st.warning(f"Live search failed ({friendly_error(e)}). Showing demo prices.")

                if not rows:
                    rows, is_demo = demo_prices(query), True

                st.session_state.rows = rows
                st.session_state.is_demo = is_demo

    rows = st.session_state.get("rows")

    if rows:
        if st.session_state.is_demo:
            st.info("Showing DEMO prices (not real). Add a SerpAPI key for live prices.")

        # ----- Filters -----
        stores = sorted({r["Store"] for r in rows})
        f1, f2 = st.columns(2)
        chosen = f1.multiselect("Filter by store", stores, default=stores)
        order = f2.radio("Sort by price", ["Low to high", "High to low"], horizontal=True)

        shown = [r for r in rows if r["Store"] in chosen]
        shown.sort(key=lambda r: r["Price (₹)"], reverse=(order == "High to low"))

        if not shown:
            st.warning("No results for the selected stores.")
        else:
            prices = [r["Price (₹)"] for r in shown]
            best = min(shown, key=lambda r: r["Price (₹)"])

            st.success(f"🏆 Best deal: {best['Store']} at ₹{best['Price (₹)']:,.0f}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Cheapest", f"₹{min(prices):,.0f}")
            m2.metric("Average", f"₹{sum(prices) / len(prices):,.0f}")
            m3.metric("You can save up to", f"₹{max(prices) - min(prices):,.0f}")

            # Results list (built without pandas)
            h1, h2, h3 = st.columns([2, 7, 2])
            h1.markdown("**Store**")
            h2.markdown("**Product (click to shop)**")
            h3.markdown("**Price**")

            for r in shown:
                c1, c2, c3 = st.columns([2, 7, 2])
                c1.write(r["Store"])
                if r["Link"]:
                    c2.link_button(r["Product"], r["Link"], use_container_width=True)
                else:
                    c2.write(r["Product"])
                c3.write(f"₹{r['Price (₹)']:,.0f}")

            st.download_button(
                "⬇️ Download results (CSV)",
                data=rows_to_csv(shown),
                file_name="pricelens_results.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption(
    "Prices come from Google Shopping via SerpAPI and can change at any time. "
    "Always confirm the final price on the store's website."
)
