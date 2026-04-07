import json
import os
import re
from pathlib import Path

NOTEBOOK_DIR = Path(r"d:\Project\10academy\Fake Image Detection\notebooks")

def summarize_markdown(source):
    if not source:
        return source
    
    # join list of lines into one string if needed
    if isinstance(source, list):
        full_text = "".join(source)
    else:
        full_text = source

    lines = full_text.split("\n")
    cleaned_lines = []
    
    # Keep headers (lines starting with #)
    # For others, if it's long, take the first 1-2 important lines.
    
    sentence_count = 0
    header_found = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1].strip(): # prevent double spacing
                cleaned_lines.append("\n")
            continue
            
        if stripped.startswith("#"):
            cleaned_lines.append(line + "\n")
            header_found = True
            continue
            
        # If it's not a header and we haven't reached our limit
        if sentence_count < 2:
            cleaned_lines.append(line + "\n")
            sentence_count += 1
            
    return cleaned_lines

def clean_notebook(path):
    print(f"Processing {path.name}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            source = cell.get("source", [])
            # Custom summaries for common patterns to ensure accuracy
            full_text = "".join(source) if isinstance(source, list) else source
            
            # Simple summarization: keep headers and first 1-2 lines of text
            lines = full_text.splitlines()
            result_lines = []
            para_lines = []
            
            for line in lines:
                if line.strip().startswith("#"):
                    if para_lines:
                        result_lines.extend(para_lines[:2]) # Keep first 2 lines of previous paragraph
                        para_lines = []
                    result_lines.append(line)
                elif line.strip():
                    para_lines.append(line)
            
            if para_lines:
                result_lines.extend(para_lines[:2])
                
            # Final output formatting
            cell["source"] = [l + "\n" for l in result_lines]
            
        new_cells.append(cell)
    
    nb["cells"] = new_cells
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Done cleaning {path.name}")

if __name__ == "__main__":
    for i in range(1, 5):
        filename = f"task{i}_"
        # Find the full filename
        matches = list(NOTEBOOK_DIR.glob(f"task{i}_*.ipynb"))
        if matches:
            clean_notebook(matches[0])
