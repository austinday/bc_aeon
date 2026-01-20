import json
from typing import Dict, Any
from collections import Counter

from .json import get_json_schema

def summarize_tabular(analyzer) -> Dict[str, Any]:
    """Summarizes a tabular file by reading a sample to infer structure and doing a fast line count."""
    try:
        import pandas as pd
        separator = '\t' if analyzer.file_extension == '.tsv' else ','
        # Perform a fast line count without reading the whole file into memory.
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract 1 for the header row

        # Read only a sample of the file to analyze structure and get a head sample.
        df_head = pd.read_csv(analyzer.file_path, sep=separator, on_bad_lines='skip', nrows=analyzer.PANDAS_HEAD_ROWS * 2)
        
        for col in df_head.select_dtypes(include=['object']):
            df_head[col] = df_head[col].apply(lambda x: (str(x)[:analyzer.MAX_CELL_LENGTH - 3] + '...') if pd.notna(x) and len(str(x)) > analyzer.MAX_CELL_LENGTH else x)
        
        return {
            "summary_type": "dataframe", "file_format": "csv" if separator == ',' else "tsv",
            "row_count": row_count,
            "column_count": len(df_head.columns),
            "columns": list(df_head.columns),
            "data_types": {col: str(dtype) for col, dtype in df_head.dtypes.items()},
            "head_sample": df_head.head(analyzer.PANDAS_HEAD_ROWS).to_dict(orient='records')
        }
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Pandas could not parse: {e}"}

def summarize_structured_text(analyzer, is_likely_structured: bool = False) -> Dict[str, Any]:
    """Summarizes a generic structured text file by sampling, avoiding reading the whole file."""
    try:
        import pandas as pd
        skiprows = 0
        if is_likely_structured:
            with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= analyzer.MAX_HEADER_SCAN_LINES: break
                    if line.strip() and not line.strip().startswith(('#', '=', '-')) and len(line.split()) > 1:
                        skiprows = i
                        break
        
        # Perform a fast line count first.
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            num_lines = sum(1 for _ in f)
        
        # Read only a sample to infer the structure and get the head.
        df_sample = pd.read_csv(analyzer.file_path, sep=None, engine='python', on_bad_lines='skip', skiprows=skiprows, comment='#', nrows=analyzer.PANDAS_HEAD_ROWS * 2)
        if len(df_sample.columns) <= 1:
            raise ValueError("File has only one column based on the initial sample.")

        df_head = df_sample.head(analyzer.PANDAS_HEAD_ROWS).copy()
        for col in df_head.select_dtypes(include=['object']):
            df_head[col] = df_head[col].apply(lambda x: (str(x)[:analyzer.MAX_CELL_LENGTH - 3] + '...') if pd.notna(x) and len(str(x)) > analyzer.MAX_CELL_LENGTH else x)
        
        return {
            "summary_type": "dataframe", "file_format": analyzer.file_extension.lstrip('.') or "structured_text",
            "row_count": num_lines - skiprows - 1,  # More accurate row count
            "column_count": len(df_head.columns), "line_count": num_lines,
            "columns": list(df_head.columns), "head_sample": df_head.to_dict(orient='records')
        }
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except Exception as e:
        raise ValueError(f"Pandas could not parse as structured text: {e}") from e

def summarize_jsonl_file(analyzer) -> Dict[str, Any]:
    try:
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            f.seek(0)
            line_count = sum(1 for _ in f)
        first_obj = json.loads(first_line)
        schema = get_json_schema(first_obj)
        return {
            "summary_type": "json_lines_summary", "file_format": "jsonl",
            "record_count": line_count,
            "schema": schema,
            "description": "JSON Lines file. Schema is from the first record."
        }
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}

def summarize_numpy_archive(analyzer) -> Dict[str, Any]:
    try:
        import numpy as np
        with np.load(analyzer.file_path) as data:
            arrays = {key: {'shape': data[key].shape, 'dtype': str(data[key].dtype)} for key in data.files}
        return {
            "summary_type": "numpy_archive_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "array_count": len(arrays),
            "arrays": arrays,
            "description": "NumPy archive file. A summary of the contained arrays, their shapes, and data types is provided."
        }
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not inspect NumPy archive: {e}"}

def summarize_tpm_file(analyzer) -> Dict[str, Any]:
    """Summarizes a TPM file by reading only the first line to get column count and a sample."""
    try:
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
        
        if not first_line:
            return {"summary_type": "empty_file", "file_format": "tpm", "description": "TPM file is empty or has an empty first line."}

        columns = first_line.split() # Splits by any whitespace
        column_count = len(columns)
        column_sample = columns[:10]

        return {
            "summary_type": "tpm_summary",
            "file_format": "tpm",
            "column_count": column_count,
            "column_sample": column_sample,
            "description": "Summary of a TPM file based on its first line. Shows total columns and a sample of the first 10."
        }
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse TPM file: {e}"}
