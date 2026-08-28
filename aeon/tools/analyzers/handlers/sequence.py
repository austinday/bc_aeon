from typing import Dict, Any
from collections import Counter

from ..limits import (
    ResourceLimitError,
    bounded_binary_readline,
    limit_error,
    read_text_prefix,
    regular_file_stat,
)


def _visit_bounded_lines(analyzer, visitor, *, max_rows=None, max_bytes=None):
    row_limit = max_rows or analyzer.MAX_RECORD_SCAN_ROWS
    byte_limit = max_bytes or analyzer.MAX_RECORD_SCAN_BYTES
    file_size = regular_file_stat(analyzer.file_path).st_size
    rows = 0
    scanned = 0
    with open(analyzer.file_path, "rb") as handle:
        while rows < row_limit and scanned < byte_limit:
            remaining = byte_limit - scanned
            if remaining < analyzer.MAX_TEXT_LINE_BYTES:
                raw = handle.readline(remaining + 1)
                if len(raw) > remaining:
                    break
            else:
                raw = bounded_binary_readline(handle, analyzer.MAX_TEXT_LINE_BYTES)
            if not raw:
                break
            if not raw.endswith((b"\n", b"\r")) and handle.tell() < file_size:
                break
            rows += 1
            scanned += len(raw)
            visitor(raw.decode("utf-8", errors="replace"))
        truncated = handle.tell() < file_size
    return rows, scanned, truncated

def summarize_record_based_data(analyzer) -> Dict[str, Any]:
    delimiters = {'.sdf': '$$$$', '.pdb': 'ENDMDL'}
    try:
        record_count = 0
        content_since_delimiter = False

        def visit(line):
            nonlocal record_count, content_since_delimiter
            stripped = line.strip()
            if not stripped:
                return
            if analyzer.file_extension == ".sdf":
                if stripped == "$$$$":
                    record_count += 1
                    content_since_delimiter = False
                else:
                    content_since_delimiter = True
            elif analyzer.file_extension == ".pdb":
                if stripped.startswith("ENDMDL"):
                    record_count += 1
                    content_since_delimiter = False
                else:
                    content_since_delimiter = True
            elif analyzer.file_extension in {".cif", ".mmcif"}:
                if stripped.startswith(("data_", "_atom_site")):
                    record_count += 1
            else:
                record_count += 1

        _rows, scanned, truncated = _visit_bounded_lines(analyzer, visit)
        if content_since_delimiter and not truncated:
            record_count += 1
        record_value = f">={record_count:,}" if truncated else record_count
        return {
            "summary_type": "structured_record_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "record_count": record_value,
            "record_delimiter": delimiters.get(analyzer.file_extension, 'N/A'),
            "scan_truncated": truncated,
            "bytes_scanned": scanned,
            "description": (
                f"Bounded metadata summary for a {analyzer.file_extension.lstrip('.')} "
                "structure file; content omitted."
            ),
        }
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}

def summarize_sequence_file(analyzer) -> Dict[str, Any]:
    """Summarizes FASTA/FASTQ-like files by reading a small chunk from the beginning."""
    headers = []
    try:
        # Read a byte-bounded prefix to avoid processing huge single-line data.
        sample_content, omitted = read_text_prefix(
            analyzer.file_path, analyzer.MAX_FASTA_SAMPLE_BYTES
        )
        
        lines = sample_content.splitlines()
        
        for line in lines:
            if line.startswith('>') or (analyzer.file_extension == '.fastq' and line.startswith('@')):
                if len(headers) < 20:  # Get a decent sample of headers
                    headers.append(line.strip())
        
        sequence_count_in_sample = len(headers)
        
        # Fallback for non-FASTA like .smi files if no headers were found
        if not headers:
             non_empty_lines = sum(1 for line in lines if line.strip())
             description = "File appears to be sequence data (e.g., SMILES), but no FASTA headers were found in the initial sample."
             return {
                "summary_type": "sequence_summary", 
                "file_format": analyzer.file_extension.lstrip('.'),
                "records_in_sample": non_empty_lines,
                "header_sample": [],
                "sample_truncated": omitted,
                "description": description
            }

        return {
            "summary_type": "sequence_summary", 
            "file_format": analyzer.file_extension.lstrip('.'),
            "sequences_in_sample": sequence_count_in_sample,
            "header_sample": headers,
            "sample_truncated": omitted,
            "description": f"A summary of the first {analyzer.MAX_FASTA_SAMPLE_BYTES:,} bytes of the file. A sample of headers is provided for FASTA/FASTQ-like files."
        }
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}

def summarize_gene_annotation(analyzer) -> Dict[str, Any]:
    """Summarizes GTF/GFF files by sampling the first N lines."""
    try:
        comments, data_sample = [], []
        feature_counts = Counter()

        def visit(line):
            if line.startswith('##'):
                if len(comments) < 20:
                    comments.append(line.strip())
                return
            if line.startswith('#') or not line.strip():
                return
            if len(data_sample) < 5:
                data_sample.append(line.strip())
            parts = line.strip().split('\t')
            if len(parts) > 2:
                feature_counts[parts[2]] += 1

        rows, scanned, truncated = _visit_bounded_lines(
            analyzer,
            visit,
            max_rows=analyzer.MAX_GENE_ANNOTATION_SAMPLE_LINES,
        )

        return {
            "summary_type": "gene_annotation_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "header_comments": comments,
            "feature_counts": dict(feature_counts.most_common(20)),
            "data_sample": "\n".join(data_sample),
            "rows_scanned": rows,
            "bytes_scanned": scanned,
            "scan_truncated": truncated,
            "description": f"Summary of the first {analyzer.MAX_GENE_ANNOTATION_SAMPLE_LINES:,} lines of a gene annotation file, showing feature counts."
        }
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}
