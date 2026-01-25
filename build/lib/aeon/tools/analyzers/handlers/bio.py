import json
from typing import Dict, Any
from collections import Counter

# Forward declarations to avoid circular imports
from .utility import summarize_opaque


def summarize_genbank(analyzer) -> Dict[str, Any]:
    """Summarizes GenBank (.gb) files by sampling records."""
    try:
        from Bio import SeqIO
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            records = list(SeqIO.parse(f, "genbank"))
        record_count = len(records)
        sample_records = records[:3] if record_count > 0 else []
        sample_summary = [
            {
                "id": str(rec.id),
                "description": rec.description[:200] + "..." if len(rec.description) > 200 else rec.description,
                "length": len(rec.seq)
            }
            for rec in sample_records
        ]
        return {
            "summary_type": "genbank_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "record_count": record_count,
            "sample_records": sample_summary,
            "description": f"GenBank file with {record_count} records. Sample of first 3 records provided."
        }
    except ImportError:
        return summarize_opaque(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse GenBank file: {e}"}


def summarize_hdf5(analyzer) -> Dict[str, Any]:
    """Summarizes HDF5 (.h5, .hdf5) files common in biological data by extracting structure."""
    try:
        import h5py
        with h5py.File(analyzer.file_path, 'r') as f:
            def extract_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    return {
                        "path": name,
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "size_bytes": obj.size * obj.dtype.itemsize
                    }
                elif isinstance(obj, h5py.Group):
                    return {
                        "path": name,
                        "type": "group",
                        "subitems": [extract_structure(child_name, child) for child_name, child in obj.items() if len([extract_structure(child_name, child)]) < 10]  # Limit depth
                    }
                return None
            structure = extract_structure('/', f)
        return {
            "summary_type": "hdf5_structure_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "file_size_bytes": analyzer.file_size,
            "structure": structure,
            "description": f"HDF5 file structure summary. Full content omitted due to size; only metadata provided."
        }
    except ImportError:
        return summarize_opaque(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not inspect HDF5 file: {e}"}
