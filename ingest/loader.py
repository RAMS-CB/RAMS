from typing import Dict, Any
from bs4 import BeautifulSoup

def extract_text(doc: Dict[str, Any]) -> str:
    """Extract and clean text from the downloaded document."""
    raw_html = doc.get("raw_html")
    if not raw_html:
        return ""
        
    soup = BeautifulSoup(raw_html, "html.parser")
    # Get text and clean it up
    text_content = soup.get_text(separator="\n", strip=True)
    return text_content