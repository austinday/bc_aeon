from typing import Dict, Any

def summarize_notebook(analyzer) -> Dict[str, Any]:
    try:
        import nbformat
        with open(analyzer.file_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        code_cells = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
        content = "\n\n# --- New Cell ---\n\n".join(code_cells)
        return {"summary_type": "notebook_code_cells", "file_format": "ipynb", "content": content}
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse notebook: {e}"}
