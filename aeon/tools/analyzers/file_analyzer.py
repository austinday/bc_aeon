import os
import re
import stat
from typing import Dict, Any

from .handlers.archive import summarize_archive
from .handlers.code import summarize_code
from .handlers.data import (
    summarize_tabular,
    summarize_structured_text,
    summarize_record_based_data,
    summarize_sequence_file,
    summarize_jsonl_file,
    summarize_numpy_archive,
    summarize_gene_annotation,
    summarize_tpm_file,
    summarize_genbank,
    summarize_hdf5
)
from .handlers.document import summarize_pdf
from .handlers.json import summarize_json
from .handlers.notebook import summarize_notebook
from .handlers.text import (
    summarize_log_file,
    analyze_generic_text,
    summarize_unrecognized_text,
)
from .handlers.utility import (
    summarize_empty,
    summarize_opaque,
    is_likely_binary,
)
from .limits import ResourceLimitError, limit_error, read_bounded_bytes

class FileAnalyzer:
    """
    Analyzes a file to produce a summary suitable for an LLM. It distinguishes
    between code files (full content) and data files (structured summary).
    This class is fully deterministic and makes no LLM calls.
    """
    # File extension categories
    CODE_EXTENSIONS = {'.py', '.sh', '.md', '.yaml', '.yml', '.toml', '.gitmodules', '.html', '.xml', '.js', '.css', '.sql', '.tex', '.ini', '.cfg', '.conf', '.properties', '.env', '.rst', '.lock', '.svg', '.http', '.rest'}
    TABULAR_EXTENSIONS = {'.csv', '.tsv', '.vcf', '.bed', '.wig', '.maf'}
    GENE_ANNOTATION_EXTENSIONS = {'.gff', '.gtf'}
    NOTEBOOK_EXTENSIONS = {'.ipynb'}
    OPAQUE_EXTENSIONS = {'.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.bin', '.onnx', '.safetensors', '.h5', '.hdf5', '.fits', '.root', '.parquet', '.feather', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.odt', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.bmp', '.mp3', '.mp4', '.mov', '.wav', '.ttf', '.otf', '.woff', '.woff2', '.eot', '.bam', '.cram', '.fast5', '.trj', '.xtc', '.dcd', '.joblib'}
    PDF_EXTENSIONS = {'.pdf'}
    STRUCTURED_RECORD_EXTENSIONS = {'.sdf', '.pdb', '.cif', '.mol2', '.gro', '.mmcif', '.pdbqt', '.ent'}
    SEQUENCE_EXTENSIONS = {'.fasta', '.fa', '.fna', '.faa', '.smi', '.fastq', '.fq', '.gb', '.gbk', '.seq', '.embl'}
    GENBANK_EXTENSIONS = {'.gb', '.gbk'}
    LOG_EXTENSIONS = {'.log'}
    JSONL_EXTENSIONS = {'.jsonl', '.jsonlines'}
    ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.gz', '.bz2', '.rar', '.7z', '.tgz'}
    NUMPY_EXTENSIONS = {'.npz', '.npy'}
    BIO_HDF5_EXTENSIONS = {'.h5', '.hdf5'}

    # Configuration
    LARGE_FILE_THRESHOLD_BYTES = 50000
    MAX_JSON_PREVIEW_SIZE = 5 * 1024
    MAX_CELL_LENGTH = 100
    PANDAS_HEAD_ROWS = 5
    TEXT_ANALYSIS_SAMPLE_LINES = 100
    STRUCTURED_TEXT_COLUMN_STD_DEV_THRESHOLD = 0.5
    STRUCTURED_TEXT_MIN_AVG_COLUMNS = 2
    BINARY_CHECK_BYTES = 2048
    NON_TEXT_CHAR_THRESHOLD = 0.3
    MAX_HEADER_SCAN_LINES = 200
    MAX_ARCHIVE_LIST_FILES = 50
    MAX_GENE_ANNOTATION_SAMPLE_LINES = 50000
    MAX_FASTA_SAMPLE_BYTES = 1 * 1024 * 1024  # 1MB
    HIDDEN_FILE_TAIL_LINES = 50

    # In-process parsing contracts.  Handlers may return bounded partial
    # metadata, but must never exceed these ceilings to obtain it.
    MAX_FULL_CONTENT_BYTES = 256 * 1024
    MAX_TEXT_PREFIX_BYTES = 64 * 1024
    MAX_TEXT_LINE_BYTES = 256 * 1024
    MAX_HIDDEN_SCAN_BYTES = 4 * 1024 * 1024
    MAX_HIDDEN_TAIL_BYTES = 64 * 1024
    MAX_JSON_PARSE_BYTES = 2 * 1024 * 1024
    MAX_NOTEBOOK_PARSE_BYTES = 8 * 1024 * 1024
    MAX_NOTEBOOK_CELLS = 2_000
    MAX_NOTEBOOK_CODE_CHARS = 250_000
    MAX_PDF_INPUT_BYTES = 64 * 1024 * 1024
    MAX_PDF_PAGES = 400
    MAX_PDF_PAGE_TEXT_CHARS = 100_000
    MAX_PDF_TEXT_CHARS = 250_000
    MAX_ARCHIVE_INPUT_BYTES = 64 * 1024 * 1024
    MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
    MAX_ARCHIVE_MEMBERS = 10_000
    MAX_ARCHIVE_EXPANDED_BYTES = 512 * 1024 * 1024
    MAX_ARCHIVE_EXPANSION_RATIO = 1_000
    MAX_ARCHIVE_STREAM_BYTES = 512 * 1024 * 1024
    MAX_ARCHIVE_NAME_BYTES = 4 * 1024
    MAX_NPY_HEADER_BYTES = 64 * 1024
    MAX_NUMPY_MEMBERS = 500
    MAX_GENBANK_INPUT_BYTES = 32 * 1024 * 1024
    MAX_GENBANK_RECORDS = 10_000
    MAX_HDF5_OBJECTS = 2_000
    MAX_HDF5_DEPTH = 16
    MAX_RECORD_SCAN_BYTES = 32 * 1024 * 1024
    MAX_RECORD_SCAN_ROWS = 500_000
    MAX_TABULAR_SAMPLE_BYTES = 2 * 1024 * 1024
    MAX_TABULAR_SAMPLE_ROWS = 50
    MAX_TABULAR_SCAN_BYTES = 64 * 1024 * 1024
    MAX_TABULAR_SCAN_ROWS = 1_000_000
    MAX_TABULAR_COLUMNS = 2_000
    MAX_JSONL_FIRST_ROW_BYTES = 1024 * 1024
    MAX_LOG_SCAN_BYTES = 64 * 1024 * 1024
    MAX_LOG_SCAN_ROWS = 1_000_000
    MAX_LOG_UNIQUE_ERRORS = 500
    MAX_LOG_SAMPLE_LINE_CHARS = 2_000

    def __init__(self, file_path: str, *, display_path: str | None = None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_stat = os.stat(file_path)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("File analyzer inputs must be regular files")
        self.file_path = file_path
        self.display_path = display_path or file_path
        self.file_size = int(file_stat.st_size)
        self.file_identity = (
            int(file_stat.st_dev),
            int(file_stat.st_ino),
            int(file_stat.st_size),
            int(file_stat.st_mtime_ns),
        )
        self.file_extension = os.path.splitext(self.display_path)[1].lower()
        self.file_name_lower = os.path.basename(self.display_path).lower()

        # Dispatch mapping from extension to handler function
        self.handler_map = {
            '.json': summarize_json,
            **{ext: summarize_code for ext in self.CODE_EXTENSIONS},
            **{ext: summarize_tabular for ext in self.TABULAR_EXTENSIONS},
            **{ext: summarize_gene_annotation for ext in self.GENE_ANNOTATION_EXTENSIONS},
            **{ext: summarize_notebook for ext in self.NOTEBOOK_EXTENSIONS},
            **{ext: summarize_opaque for ext in self.OPAQUE_EXTENSIONS},
            **{ext: summarize_pdf for ext in self.PDF_EXTENSIONS},
            **{ext: summarize_record_based_data for ext in self.STRUCTURED_RECORD_EXTENSIONS},
            **{ext: summarize_sequence_file for ext in self.SEQUENCE_EXTENSIONS},
            **{ext: summarize_genbank for ext in self.GENBANK_EXTENSIONS},
            **{ext: summarize_log_file for ext in self.LOG_EXTENSIONS},
            **{ext: summarize_jsonl_file for ext in self.JSONL_EXTENSIONS},
            **{ext: summarize_archive for ext in self.ARCHIVE_EXTENSIONS},
            **{ext: summarize_numpy_archive for ext in self.NUMPY_EXTENSIONS},
            **{ext: summarize_hdf5 for ext in self.BIO_HDF5_EXTENSIONS},
        }

    def identity_is_current(self) -> bool:
        try:
            current = os.stat(self.file_path)
        except OSError:
            return False
        return self.file_identity == (
            int(current.st_dev),
            int(current.st_ino),
            int(current.st_size),
            int(current.st_mtime_ns),
        )

    def _summarize_hidden_file(self) -> Dict[str, Any]:
        """Special summarizer for hidden files: provide only the last 50 lines for text files."""
        if is_likely_binary(self):
            return summarize_opaque(self)
        
        try:
            if self.file_size <= self.MAX_HIDDEN_SCAN_BYTES:
                raw = read_bounded_bytes(
                    self.file_path,
                    self.MAX_HIDDEN_SCAN_BYTES,
                    label="hidden text file",
                )
                lines = raw.decode("utf-8", errors="replace").splitlines(keepends=True)
                num_lines: int | str = len(lines)
                tail_lines = lines[-self.HIDDEN_FILE_TAIL_LINES:]
                content_tail = ''.join(tail_lines)
                line_description = f"{num_lines} total lines"
            else:
                # A reverse byte window is enough for a useful tail and avoids a
                # full-file line count.  Drop the first fragment when the window
                # starts in the middle of a line.
                with open(self.file_path, "rb") as handle:
                    handle.seek(-self.MAX_HIDDEN_TAIL_BYTES, os.SEEK_END)
                    raw = handle.read(self.MAX_HIDDEN_TAIL_BYTES)
                text = raw.decode("utf-8", errors="replace")
                first_break = text.find("\n")
                if first_break >= 0:
                    text = text[first_break + 1:]
                tail_lines = text.splitlines(keepends=True)[-self.HIDDEN_FILE_TAIL_LINES:]
                content_tail = ''.join(tail_lines)
                num_lines = "not counted (scan limit)"
                line_description = "total line count omitted by scan limit"

            return {
                "summary_type": "hidden_file_tail",
                "file_format": self.file_extension.lstrip('.') or 'hidden',
                "content": content_tail,
                "total_lines": num_lines,
                "description": (
                    f"Hidden file ({os.path.basename(self.display_path)}). Full content omitted; "
                    f"only the last {len(tail_lines)} bounded lines are provided "
                    f"({line_description})."
                ),
            }
        except Exception as e:
            return {"summary_type": "error", "error_message": f"Could not read hidden file: {e}"}

    def analyze(self) -> Dict[str, Any]:
        special_files = ['requirements.txt', '.gitignore']
        summary = None
        if self.file_size == 0:
            summary = summarize_empty(self)
        elif self.file_name_lower.startswith('.'):  # Handle hidden files specially
            summary = self._summarize_hidden_file()
        elif 'readme' in self.file_name_lower or self.file_name_lower in special_files:
            summary = summarize_code(self)
        else:
            # Special case: Large file with no extension. Try to infer type.
            if self.file_size > self.LARGE_FILE_THRESHOLD_BYTES and not self.file_extension:
                name_parts = re.split(r'[._-]', self.file_name_lower)
                if len(name_parts) > 1 and name_parts[-1] == 'tpm':
                    summary = summarize_tpm_file(self)
            
            if summary is None: # If special handling did not apply
                handler = self.handler_map.get(self.file_extension)
                if handler:
                    summary = handler(self)
                elif is_likely_binary(self):
                    summary = summarize_opaque(self)
                else:
                    summary = analyze_generic_text(self)
                    if summary.get("summary_type") == "_structured_text_internal":
                        try:
                            summary = summarize_structured_text(self, is_likely_structured=summary["is_likely_structured"])
                        except ValueError:
                            try:
                                summary = summarize_unrecognized_text(self)
                            except ResourceLimitError as exc:
                                summary = limit_error(exc)

        base_info = {
            "file_name": os.path.basename(self.display_path),
            "file_size_bytes": self.file_size
        }
        return {**base_info, **summary}
