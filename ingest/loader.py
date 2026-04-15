from typing import Dict, Any

def extract_text(doc: Dict[str, Any]) -> str:
    """Extract the markdown text from the downloaded document."""
    raw_content = doc.get("raw_html")
    if not raw_content:
        return ""
        
    # Since we now use Jina Reader API, raw_html is actually clean Markdown text.
    # No HTML parsing with BeautifulSoup is needed anymore.
    return str(raw_content).strip()