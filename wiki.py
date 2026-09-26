"""Fetch English Wikipedia articles and map them to the model's input fields.

Keep in sync with web/lib/Wikipedia.php so the script and the website give identical predictions:
    title    = article title
    keywords = short description + non-hidden categories (one per line)
    abstract = plain-text lead section, or the whole article when scope == "full"
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "COSIPredictor/1.0 (https://cosipredictor.dandeblasio.com)"


class WikipediaError(Exception):
    pass


def parse_title(text):
    """Article title from a title or an en.wikipedia.org URL."""
    text = text.strip()
    m = re.match(r"^(?:https?://)?([a-z0-9-]+)(?:\.m)?\.wikipedia\.org/"
                 r"(?:wiki/([^?#]+)|w/index\.php\?(?:.*&)?title=([^&#]+))", text, re.IGNORECASE)
    if m:
        if m.group(1).lower() != "en":
            raise WikipediaError("Only English Wikipedia articles are supported.")
        text = urllib.parse.unquote(m.group(2) or m.group(3))
    title = text.replace("_", " ").strip()
    if not title:
        raise WikipediaError("Empty article title.")
    return title


def fetch(title_or_url, scope="lead", retries=3, timeout=15):
    title = parse_title(title_or_url)
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "prop": "extracts|categories|info|pageprops|description",
        "explaintext": "1", "clshow": "!hidden", "cllimit": "max",
        "inprop": "url", "ppprop": "disambiguation", "titles": title,
    }
    if scope != "full":
        params["exintro"] = "1"
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise WikipediaError(f"Could not reach Wikipedia: {e}") from e
            time.sleep(2 ** attempt)

    pages = data.get("query", {}).get("pages", [])
    page = pages[0] if pages else None
    if not page or page.get("invalid"):
        raise WikipediaError(f"'{title}' is not a valid Wikipedia title.")
    if page.get("missing"):
        raise WikipediaError(f"No English Wikipedia article is titled '{title}'.")
    if "disambiguation" in page.get("pageprops", {}):
        raise WikipediaError(f"'{page['title']}' is a disambiguation page.")
    redirects = data["query"].get("redirects") or [{}]
    return {
        "title": page["title"],
        "url": page.get("fullurl", ""),
        "description": page.get("description", ""),
        "categories": [re.sub(r"^Category:", "", c["title"]) for c in page.get("categories", [])],
        "text": page.get("extract", "").strip(),
        "redirected_from": redirects[0].get("from"),
    }


def to_fields(article):
    keywords = [k for k in [article["description"], *article["categories"]] if k]
    return {"title": article["title"], "keywords": "\n".join(keywords), "abstract": article["text"]}
