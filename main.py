import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

BASE_URL = "https://namu.wiki/w/"
MAX_LINK_DEPTH = 1
MAX_WORKERS = 5
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/104.0.5112.79 Safari/537.36"
}
AD_PATTERN = re.compile(
    r"(^|[-_\s])(ad|ads|advert|advertisement|banner|sponsor|sponsored|promoted|\uad11\uace0)([-_\s]|$)",
    re.IGNORECASE,
)
BRACKET_LINK_PATTERN = re.compile(r"\[\[([^\]|#]+)")


def safe_filename(name):
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return name or "document"


def is_ad_tag(tag):
    check_values = []
    for attr in ("id", "class", "aria-label", "role"):
        value = tag.get(attr)
        if isinstance(value, list):
            check_values.extend(value)
        elif value:
            check_values.append(str(value))

    for attr, value in tag.attrs.items():
        if attr.startswith("data-"):
            if isinstance(value, list):
                check_values.extend(value)
            elif value:
                check_values.append(str(value))

    return any(AD_PATTERN.search(value) for value in check_values)


def remove_noise_tags(html):
    for tag in html(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    for tag in list(html.find_all(is_ad_tag)):
        tag.decompose()


def extract_document_id_from_href(href):
    parsed = urlparse(href)
    path = parsed.path if parsed.scheme else href.split("?", 1)[0].split("#", 1)[0]

    if not path.startswith("/w/"):
        return ""

    document_id = unquote(path[len("/w/"):]).strip()
    return document_id


def extract_linked_document_ids(html, page_source):
    linked_document_ids = set()

    for link in html.find_all("a", href=True):
        document_id = extract_document_id_from_href(link["href"])
        if document_id:
            linked_document_ids.add(document_id)

    for match in BRACKET_LINK_PATTERN.finditer(page_source):
        document_id = match.group(1).strip()
        if document_id:
            linked_document_ids.add(document_id)

    return linked_document_ids


def clean_text(text):
    return " ".join(text.split())


def extract_structured_text(node):
    if isinstance(node, NavigableString):
        return clean_text(str(node))

    if not getattr(node, "name", None):
        return ""

    name = node.name.lower()

    if name in {"script", "style", "noscript"}:
        return ""

    if name == "br":
        return "\n"

    if name in {"td", "th"}:
        return extract_children_text(node).strip()

    if name == "tr":
        cells = [
            extract_structured_text(cell).strip()
            for cell in node.find_all(["td", "th"], recursive=False)
        ]
        return " | ".join(cell for cell in cells if cell) + "\n"

    if name == "li":
        text = extract_children_text(node).strip()
        return f"- {text}\n" if text else ""

    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        text = extract_children_text(node).strip()
        return f"\n\n{text}\n" if text else ""

    if name in {"p", "blockquote", "table", "ul", "ol"}:
        text = extract_children_text(node).strip()
        return f"{text}\n\n" if text else ""

    return extract_children_text(node)


def extract_children_text(node):
    parts = []
    for child in node.children:
        text = extract_structured_text(child)
        if not text:
            continue

        if parts and not parts[-1].endswith(("\n", " ", "| ")) and not text.startswith(("\n", " ", "|")):
            parts.append(" ")
        parts.append(text)

    return "".join(parts)


def normalize_text(text):
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def parse_document(document_id):
    response = requests.get(BASE_URL + quote(document_id, safe=""), headers=HEADERS)
    response.raise_for_status()

    html = BeautifulSoup(response.text, "html.parser")
    remove_noise_tags(html)
    linked_document_ids = extract_linked_document_ids(html, response.text)
    linked_document_ids.discard(document_id)

    content = html.find("article") or html.find("main") or html.body or html
    text = normalize_text(extract_structured_text(content))

    filename = safe_filename(document_id)
    with open(os.path.join("page_sources", f"{filename}.html"), "w", encoding="utf-8") as file:
        file.write(str(html))

    with open(os.path.join("parsed_texts", f"{filename}.txt"), "w", encoding="utf-8") as file:
        file.write(text)

    return linked_document_ids


def parse_documents(document_ids):
    scheduled = set(document_ids)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(parse_document, document_id): (document_id, 0)
            for document_id in document_ids
        }

        while futures:
            for future in as_completed(list(futures)):
                document_id, depth = futures.pop(future)

                try:
                    linked_document_ids = future.result()
                except requests.RequestException as error:
                    print(f"Failed: {document_id} ({error})")
                    continue

                print(f"Saved: {document_id}")

                if depth >= MAX_LINK_DEPTH:
                    continue

                for linked_document_id in sorted(linked_document_ids):
                    if linked_document_id in scheduled:
                        continue

                    scheduled.add(linked_document_id)
                    futures[executor.submit(parse_document, linked_document_id)] = (
                        linked_document_id,
                        depth + 1,
                    )


os.makedirs("page_sources", exist_ok=True)
os.makedirs("parsed_texts", exist_ok=True)

document_ids = [document_id.strip() for document_id in input("Enter ID: ").split(",")]
document_ids = [document_id for document_id in document_ids if document_id]

parse_documents(document_ids)
