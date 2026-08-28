import io
import json
from typing import Dict, Any
import zipfile

from .json import get_json_schema
from .archive import (
    _validate_expansion,
    _validate_name,
    _zip_preflight,
)
from ..limits import (
    ResourceLimitError,
    bounded_binary_readline,
    limit_error,
    regular_file_stat,
    require_max_file_bytes,
    scan_text_lines,
)


def _bounded_row_count(analyzer):
    return scan_text_lines(
        analyzer.file_path,
        max_rows=analyzer.MAX_TABULAR_SCAN_ROWS,
        max_bytes=analyzer.MAX_TABULAR_SCAN_BYTES,
        max_line_bytes=analyzer.MAX_TEXT_LINE_BYTES,
    )


def _bounded_table_sample(analyzer):
    scan = scan_text_lines(
        analyzer.file_path,
        max_rows=analyzer.MAX_TABULAR_SAMPLE_ROWS,
        max_bytes=analyzer.MAX_TABULAR_SAMPLE_BYTES,
        max_line_bytes=analyzer.MAX_TEXT_LINE_BYTES,
        sample_rows=analyzer.MAX_TABULAR_SAMPLE_ROWS,
    )
    if not scan.sampled_lines:
        raise ValueError("file has no complete rows in the bounded sample")
    return list(scan.sampled_lines)


def _bounded_count_value(scan, *, header_rows: int = 0):
    count = max(0, scan.row_count - header_rows)
    return f">={count:,}" if scan.truncated else count


def _validate_dataframe_width(analyzer, dataframe) -> None:
    columns = len(dataframe.columns)
    if columns > analyzer.MAX_TABULAR_COLUMNS:
        raise ResourceLimitError(
            f"table has {columns:,} sampled columns; limit is "
            f"{analyzer.MAX_TABULAR_COLUMNS:,} columns"
        )

def summarize_tabular(analyzer) -> Dict[str, Any]:
    """Summarize a table from byte/line/row-bounded input only."""
    try:
        import pandas as pd
        separator = '\t' if analyzer.file_extension == '.tsv' else ','
        row_scan = _bounded_row_count(analyzer)
        sample_lines = _bounded_table_sample(analyzer)
        if sample_lines[0].count(separator) + 1 > analyzer.MAX_TABULAR_COLUMNS:
            raise ResourceLimitError(
                f"table header exceeds the {analyzer.MAX_TABULAR_COLUMNS:,}-column limit"
            )
        df_head = pd.read_csv(
            io.StringIO("".join(sample_lines)),
            sep=separator,
            on_bad_lines='skip',
            nrows=analyzer.PANDAS_HEAD_ROWS * 2,
        )
        _validate_dataframe_width(analyzer, df_head)
        
        for col in df_head.select_dtypes(include=['object']):
            df_head[col] = df_head[col].apply(lambda x: (str(x)[:analyzer.MAX_CELL_LENGTH - 3] + '...') if pd.notna(x) and len(str(x)) > analyzer.MAX_CELL_LENGTH else x)
        
        return {
            "summary_type": "dataframe", "file_format": "csv" if separator == ',' else "tsv",
            "row_count": _bounded_count_value(row_scan, header_rows=1),
            "row_scan_truncated": row_scan.truncated,
            "column_count": len(df_head.columns),
            "columns": list(df_head.columns),
            "data_types": {col: str(dtype) for col, dtype in df_head.dtypes.items()},
            "head_sample": df_head.head(analyzer.PANDAS_HEAD_ROWS).to_dict(orient='records')
        }
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Pandas could not parse: {e}"}

def summarize_structured_text(analyzer, is_likely_structured: bool = False) -> Dict[str, Any]:
    """Summarizes a generic structured text file by sampling, avoiding reading the whole file."""
    try:
        import pandas as pd
        sample_lines = _bounded_table_sample(analyzer)
        skiprows = 0
        if is_likely_structured:
            for i, line in enumerate(sample_lines[:analyzer.MAX_HEADER_SCAN_LINES]):
                if line.strip() and not line.strip().startswith(('#', '=', '-')) and len(line.split()) > 1:
                    skiprows = i
                    break

        row_scan = _bounded_row_count(analyzer)
        df_sample = pd.read_csv(
            io.StringIO("".join(sample_lines)),
            sep=None,
            engine='python',
            on_bad_lines='skip',
            skiprows=skiprows,
            comment='#',
            nrows=analyzer.PANDAS_HEAD_ROWS * 2,
        )
        _validate_dataframe_width(analyzer, df_sample)
        if len(df_sample.columns) <= 1:
            raise ValueError("File has only one column based on the initial sample.")

        df_head = df_sample.head(analyzer.PANDAS_HEAD_ROWS).copy()
        for col in df_head.select_dtypes(include=['object']):
            df_head[col] = df_head[col].apply(lambda x: (str(x)[:analyzer.MAX_CELL_LENGTH - 3] + '...') if pd.notna(x) and len(str(x)) > analyzer.MAX_CELL_LENGTH else x)
        
        return {
            "summary_type": "dataframe", "file_format": analyzer.file_extension.lstrip('.') or "structured_text",
            "row_count": _bounded_count_value(row_scan, header_rows=skiprows + 1),
            "column_count": len(df_head.columns),
            "line_count": (f">={row_scan.row_count:,}" if row_scan.truncated else row_scan.row_count),
            "row_scan_truncated": row_scan.truncated,
            "columns": list(df_head.columns), "head_sample": df_head.to_dict(orient='records')
        }
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except ResourceLimitError as e:
        raise ValueError(f"Resource limit: {e}") from e
    except Exception as e:
        raise ValueError(f"Pandas could not parse as structured text: {e}") from e

def summarize_jsonl_file(analyzer) -> Dict[str, Any]:
    try:
        scan = scan_text_lines(
            analyzer.file_path,
            max_rows=analyzer.MAX_TABULAR_SCAN_ROWS,
            max_bytes=analyzer.MAX_TABULAR_SCAN_BYTES,
            max_line_bytes=analyzer.MAX_JSONL_FIRST_ROW_BYTES,
            sample_rows=1,
        )
        if not scan.sampled_lines:
            raise ValueError("JSON Lines file has no complete first record")
        first_obj = json.loads(scan.sampled_lines[0])
        schema = get_json_schema(first_obj)
        return {
            "summary_type": "json_lines_summary", "file_format": "jsonl",
            "record_count": (f">={scan.row_count:,}" if scan.truncated else scan.row_count),
            "row_scan_truncated": scan.truncated,
            "schema": schema,
            "description": "JSON Lines file. Schema is from the first record."
        }
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}

def summarize_numpy_archive(analyzer) -> Dict[str, Any]:
    try:
        import numpy as np

        def read_header(handle):
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    handle, max_header_size=analyzer.MAX_NPY_HEADER_BYTES
                )
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    handle, max_header_size=analyzer.MAX_NPY_HEADER_BYTES
                )
            else:
                raise ResourceLimitError(
                    f"unsupported NumPy header version {version!r}"
                )
            return {
                "shape": tuple(int(dimension) for dimension in shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
            }

        arrays = {}
        if analyzer.file_extension == ".npy":
            regular_file_stat(analyzer.file_path)
            with open(analyzer.file_path, "rb") as handle:
                arrays[analyzer.file_name_lower] = read_header(handle)
            total_arrays = 1
        else:
            file_size = require_max_file_bytes(
                analyzer.file_path,
                analyzer.MAX_ARCHIVE_INPUT_BYTES,
                label="NumPy archive",
            )
            expected_count = _zip_preflight(analyzer, file_size)
            if expected_count > analyzer.MAX_NUMPY_MEMBERS:
                raise ResourceLimitError(
                    f"NumPy archive has {expected_count:,} members; limit is "
                    f"{analyzer.MAX_NUMPY_MEMBERS:,} members"
                )
            with zipfile.ZipFile(analyzer.file_path, "r") as archive:
                members = archive.infolist()
                if len(members) != expected_count:
                    raise ResourceLimitError(
                        "NumPy archive member count changed during inspection"
                    )
                expanded = 0
                compressed = 0
                array_members = []
                for member in members:
                    _validate_name(analyzer, member.filename)
                    expanded += int(member.file_size)
                    compressed += int(member.compress_size)
                    _validate_expansion(
                        analyzer,
                        compressed_bytes=max(compressed, 1),
                        expanded_bytes=expanded,
                    )
                    if member.filename.endswith(".npy"):
                        array_members.append(member)
                total_arrays = len(array_members)
                for member in array_members[:50]:
                    with archive.open(member, "r") as handle:
                        arrays[member.filename[:-4]] = read_header(handle)
        return {
            "summary_type": "numpy_archive_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "array_count": total_arrays,
            "arrays": arrays,
            "description": (
                "NumPy array metadata read from bounded headers only; array payloads "
                "were not materialized."
            ),
        }
    except ImportError:
        from .utility import summarize_opaque
        return summarize_opaque(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not inspect NumPy archive: {e}"}

def summarize_tpm_file(analyzer) -> Dict[str, Any]:
    """Summarizes a TPM file by reading only the first line to get column count and a sample."""
    try:
        with open(analyzer.file_path, "rb") as handle:
            first_line = bounded_binary_readline(
                handle, analyzer.MAX_TEXT_LINE_BYTES
            ).decode("utf-8", errors="replace").strip()
        
        if not first_line:
            return {"summary_type": "empty_file", "file_format": "tpm", "description": "TPM file is empty or has an empty first line."}

        columns = first_line.split() # Splits by any whitespace
        column_count = len(columns)
        if column_count > analyzer.MAX_TABULAR_COLUMNS:
            raise ResourceLimitError(
                f"TPM header has {column_count:,} columns; limit is "
                f"{analyzer.MAX_TABULAR_COLUMNS:,} columns"
            )
        column_sample = columns[:10]

        return {
            "summary_type": "tpm_summary",
            "file_format": "tpm",
            "column_count": column_count,
            "column_sample": column_sample,
            "description": "Summary of a TPM file based on its first line. Shows total columns and a sample of the first 10."
        }
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse TPM file: {e}"}
