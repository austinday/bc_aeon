import os
from typing import Dict, Any

from ..limits import ResourceLimitError, limit_error, require_max_file_bytes


def summarize_pdf(analyzer) -> Dict[str, Any]:
    try:
        require_max_file_bytes(
            analyzer.file_path,
            analyzer.MAX_PDF_INPUT_BYTES,
            label="PDF",
        )
        import fitz  # PyMuPDF

        full_text_list = []
        total_chars = 0
        with fitz.open(analyzer.file_path) as doc:
            page_count = int(doc.page_count)
            if page_count > analyzer.MAX_PDF_PAGES:
                raise ResourceLimitError(
                    f"PDF has {page_count:,} pages; limit is "
                    f"{analyzer.MAX_PDF_PAGES:,} pages"
                )
            for page_number in range(page_count):
                text = doc.load_page(page_number).get_text("text").strip()
                if len(text) > analyzer.MAX_PDF_PAGE_TEXT_CHARS:
                    raise ResourceLimitError(
                        f"PDF page {page_number + 1} produced {len(text):,} text "
                        f"characters; per-page limit is "
                        f"{analyzer.MAX_PDF_PAGE_TEXT_CHARS:,}"
                    )
                if text.lower().startswith(('references', 'bibliography')):
                    continue
                total_chars += len(text)
                if total_chars > analyzer.MAX_PDF_TEXT_CHARS:
                    raise ResourceLimitError(
                        f"PDF text exceeds the {analyzer.MAX_PDF_TEXT_CHARS:,}-character limit"
                    )
                full_text_list.append(text)
        full_text = "\n\n".join(full_text_list)

        return {"summary_type": "full_content", "file_format": "pdf_text", "content": full_text}
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"PyMuPDF could not parse PDF '{os.path.basename(analyzer.file_path)}': {e}"}
