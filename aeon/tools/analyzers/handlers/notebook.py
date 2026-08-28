import json
from typing import Dict, Any

from ..limits import ResourceLimitError, limit_error, read_bounded_bytes


def summarize_notebook(analyzer) -> Dict[str, Any]:
    try:
        raw = read_bounded_bytes(
            analyzer.file_path,
            analyzer.MAX_NOTEBOOK_PARSE_BYTES,
            label="notebook",
        )
        notebook = json.loads(raw.decode("utf-8"))
        cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
        if not isinstance(cells, list):
            raise ValueError("notebook cells must be a list")
        if len(cells) > analyzer.MAX_NOTEBOOK_CELLS:
            raise ResourceLimitError(
                f"notebook has {len(cells):,} cells; limit is "
                f"{analyzer.MAX_NOTEBOOK_CELLS:,} cells"
            )

        code_cells = []
        code_chars = 0
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(str(part) for part in source)
            elif not isinstance(source, str):
                source = str(source)
            code_chars += len(source)
            if code_chars > analyzer.MAX_NOTEBOOK_CODE_CHARS:
                raise ResourceLimitError(
                    f"notebook code has more than {analyzer.MAX_NOTEBOOK_CODE_CHARS:,} "
                    "characters"
                )
            code_cells.append(source)
        content = "\n\n# --- New Cell ---\n\n".join(code_cells)
        return {"summary_type": "notebook_code_cells", "file_format": "ipynb", "content": content}
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse notebook: {e}"}
