import os
import re
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

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.file_extension = os.path.splitext(file_path)[1].lower()
        self.file_name_lower = os.path.basename(file_path).lower()

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
            **{ext: summarize_log_file for ext in self.LOG_EXTENSIONS},
            **{ext: summarize_jsonl_file for ext in self.JSONL_EXTENSIONS},
            **{ext: summarize_archive for ext in self.ARCHIVE_EXTENSIONS},
            **{ext: summarize_numpy_archive for ext in self.NUMPY_EXTENSIONS},
            **{ext: summarize_hdf5 for ext in self.BIO_HDF5_EXTENSIONS},
        }

    def _summarize_hidden_file(self) -> Dict[str, Any]:
        """Special summarizer for hidden files: provide only the last 50 lines for text files."""
        if is_likely_binary(self):
            return summarize_opaque(self)
        
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            num_lines = len(lines)
            tail_lines = lines[-self.HIDDEN_FILE_TAIL_LINES:] if len(lines) > self.HIDDEN_FILE_TAIL_LINES else lines
            content_tail = ''.join(tail_lines)
            
            return {
                "summary_type": "hidden_file_tail",
                "file_format": self.file_extension.lstrip('.') or 'hidden',
                "content": content_tail,
                "total_lines": num_lines,
                "description": f"Hidden file ({os.path.basename(self.file_path)}). Full content omitted; only last {min(self.HIDDEN_FILE_TAIL_LINES, num_lines)} lines provided ({num_lines} total lines)."
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
                            summary = summarize_unrecognized_text(self)

        base_info = {
            "file_name": os.path.basename(self.file_path),
            "file_size_bytes": self.file_size
        }
        return {**base_info, **summary}
