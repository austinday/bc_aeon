import os
from typing import Dict, Any

def summarize_pdf(analyzer) -> Dict[str, Any]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(analyzer.file_path)
        full_text_list = [page.get_text("text").strip() for page in doc if not page.get_text("text").strip().lower().startswith(('references', 'bibliography'))]
        doc.close()
        full_text = "\n\n".join(full_text_list)
        
        return {"summary_type": "full_content", "file_format": "pdf_text", "content": full_text}
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"PyMuPDF could not parse PDF '{os.path.basename(analyzer.file_path)}': {e}"}
