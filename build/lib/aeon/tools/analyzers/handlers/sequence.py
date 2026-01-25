import json
from typing import Dict, Any
from collections import Counter
import re

def summarize_record_based_data(analyzer) -> Dict[str, Any]:
    delimiters = {'.sdf': '$$$$', '.pdb': 'ENDMDL'}
    # For new protein structure formats, use generic line-based counting or known delimiters if available
    # .cif uses 'data_' sections, but for simplicity, count lines as approximate records
    # Omit content_sample for all structure files to avoid including complex data
    protein_structure_extensions = {'.sdf', '.pdb', '.cif', '.mol2', '.gro', '.mmcif', '.pdbqt', '.ent'}
    try:
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if analyzer.file_extension in protein_structure_extensions:
            if analyzer.file_extension in delimiters:
                delimiter = delimiters[analyzer.file_extension]
                records = [r for r in content.split(delimiter) if r.strip()]
                record_count = len(records)
            else:
                # Generic counting for .cif, .mol2, etc. (e.g., lines or sections)
                if analyzer.file_extension == '.cif':
                    # Approximate records by counting 'data_' or '_atom_site' lines
                    record_count = len(re.findall(r'^data_|^_atom_site', content, re.MULTILINE))
                else:
                    # Fallback: approximate by non-empty lines or fixed estimate
                    record_count = sum(1 for line in content.splitlines() if line.strip())
            return {
                "summary_type": "structured_record_summary", "file_format": analyzer.file_extension.lstrip('.'),
                "record_count": record_count, "record_delimiter": delimiters.get(analyzer.file_extension, 'N/A'),
                "description": f"Complex protein structure file ({analyzer.file_extension.lstrip('.')}). Content omitted to save context space; only metadata provided."
            }
        else:
            # Original logic for non-structure record files
            delimiter = delimiters.get(analyzer.file_extension, '$$$$')
            records = [r for r in content.split(delimiter) if r.strip()]
            sample_content = delimiter.join(records[:2])
            if len(records) > 2:
                sample_content += f'\n{delimiter} ...and more'
            return {
                "summary_type": "structured_record_summary", "file_format": analyzer.file_extension.lstrip('.'),
                "record_count": len(records), "record_delimiter": delimiter, "content_sample": sample_content,
                "description": f"File with record-based data. A sample of the first 2 records is provided."
            }
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}

def summarize_sequence_file(analyzer) -> Dict[str, Any]:
    """Summarizes FASTA/FASTQ-like files by reading a small chunk from the beginning."""
    headers = []
    try:
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read a sample chunk to avoid processing huge files with single long lines
            sample_content = f.read(analyzer.MAX_FASTA_SAMPLE_BYTES)
        
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
                "description": description
            }

        return {
            "summary_type": "sequence_summary", 
            "file_format": analyzer.file_extension.lstrip('.'),
            "sequences_in_sample": sequence_count_in_sample,
            "header_sample": headers,
            "description": f"A summary of the first {analyzer.MAX_FASTA_SAMPLE_BYTES:,} bytes of the file. A sample of headers is provided for FASTA/FASTQ-like files."
        }
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}

def summarize_gene_annotation(analyzer) -> Dict[str, Any]:
    """Summarizes GTF/GFF files by sampling the first N lines."""
    try:
        comments, data_sample = [], []
        feature_counts = Counter()
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= analyzer.MAX_GENE_ANNOTATION_SAMPLE_LINES:
                    break
                if line.startswith('##'):
                    if len(comments) < 20:
                        comments.append(line.strip())
                    continue
                if line.startswith('#') or not line.strip():
                    continue
                if len(data_sample) < 5:
                    data_sample.append(line.strip())
                
                parts = line.strip().split('\t')
                if len(parts) > 2:
                    feature_counts[parts[2]] += 1

        return {
            "summary_type": "gene_annotation_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "header_comments": comments,
            "feature_counts": dict(feature_counts.most_common(20)),
            "data_sample": "\n".join(data_sample),
            "description": f"Summary of the first {analyzer.MAX_GENE_ANNOTATION_SAMPLE_LINES:,} lines of a gene annotation file, showing feature counts."
        }
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}
