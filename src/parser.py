import os
import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_sec_documents(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Safely load HTML files, clean them by stripping script/style/nav tags,
    and extract text nodes with structural metadata to a structured JSON file.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Ensure processed directory exists
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Safely find .html and .htm files
    html_files = list(raw_path.glob("*.html")) + list(raw_path.glob("*.htm"))
    
    if not html_files:
        logger.warning(f"No HTML files found in {raw_dir}")
        return

    # Process each file with a progress bar
    for file_path in tqdm(html_files, desc="Parsing SEC Documents"):
        company = file_path.stem
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            # Clean HTML by stripping script, style, nav tags
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
                
            extracted_data = []
            current_section = "Document Start"
            
            # Traverse sequentially, extracting allowed tags
            # To avoid text duplication in nested elements (like table -> tr -> td),
            # we extract text primarily from headers, paragraphs, and table data cells.
            for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "table", "tr", "td"]):
                tag_name = element.name
                
                if tag_name in ["h1", "h2", "h3", "h4"]:
                    text = element.get_text(separator=" ", strip=True)
                    if text:
                        current_section = text
                        extracted_data.append({
                            "company": company,
                            "section": current_section,
                            "tag_type": tag_name,
                            "text_content": text
                        })
                elif tag_name == "p":
                    text = element.get_text(separator=" ", strip=True)
                    if text:
                        extracted_data.append({
                            "company": company,
                            "section": current_section,
                            "tag_type": tag_name,
                            "text_content": text
                        })
                elif tag_name == "td":
                    # Extract cell text to preserve table data without duplicating row/table text
                    text = element.get_text(separator=" ", strip=True)
                    if text:
                        extracted_data.append({
                            "company": company,
                            "section": current_section,
                            "tag_type": tag_name,
                            "text_content": text
                        })

            # Output unified structured JSON
            out_file = processed_path / f"{company}.json"
            with open(out_file, "w", encoding="utf-8") as out_f:
                json.dump(extracted_data, out_f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    parse_sec_documents()
