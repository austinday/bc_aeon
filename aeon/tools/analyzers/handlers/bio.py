from typing import Dict, Any

# Forward declarations to avoid circular imports
from .utility import summarize_opaque
from ..limits import (
    ResourceLimitError,
    limit_error,
    regular_file_stat,
    require_max_file_bytes,
)


def summarize_genbank(analyzer) -> Dict[str, Any]:
    """Summarizes GenBank (.gb) files by sampling records."""
    try:
        require_max_file_bytes(
            analyzer.file_path,
            analyzer.MAX_GENBANK_INPUT_BYTES,
            label="GenBank file",
        )
        from Bio import SeqIO
        record_count = 0
        sample_records = []
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for record in SeqIO.parse(f, "genbank"):
                record_count += 1
                if record_count > analyzer.MAX_GENBANK_RECORDS:
                    raise ResourceLimitError(
                        f"GenBank file has more than {analyzer.MAX_GENBANK_RECORDS:,} records"
                    )
                if len(sample_records) < 3:
                    description = str(record.description)
                    sample_records.append({
                        "id": str(record.id)[:200],
                        "description": (
                            description[:200] + "..."
                            if len(description) > 200
                            else description
                        ),
                        "length": len(record.seq),
                    })
        return {
            "summary_type": "genbank_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "record_count": record_count,
            "sample_records": sample_records,
            "description": f"GenBank file with {record_count} records. Sample of first 3 records provided."
        }
    except ImportError:
        return summarize_opaque(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse GenBank file: {e}"}


def summarize_hdf5(analyzer) -> Dict[str, Any]:
    """Summarizes HDF5 (.h5, .hdf5) files common in biological data by extracting structure."""
    try:
        regular_file_stat(analyzer.file_path)
        import h5py
        with h5py.File(analyzer.file_path, 'r') as f:
            objects = []
            limit_hit = False

            def extract_structure(name, obj):
                nonlocal limit_hit
                if len(objects) >= analyzer.MAX_HDF5_OBJECTS:
                    limit_hit = True
                    return "aeon-object-limit"
                depth = 0 if not name else name.count("/") + 1
                if depth > analyzer.MAX_HDF5_DEPTH:
                    limit_hit = True
                    return "aeon-depth-limit"
                name_bytes = len(name.encode("utf-8", errors="replace"))
                if name_bytes > analyzer.MAX_ARCHIVE_NAME_BYTES:
                    limit_hit = True
                    return "aeon-name-limit"
                if isinstance(obj, h5py.Dataset):
                    entry = {
                        "path": name,
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "size_bytes": obj.size * obj.dtype.itemsize,
                    }
                elif isinstance(obj, h5py.Group):
                    entry = {
                        "path": name,
                        "type": "group",
                        "child_count": len(obj),
                    }
                else:
                    entry = {"path": name, "type": type(obj).__name__}
                objects.append(entry)
                return None

            result = f.visititems(extract_structure)
            if limit_hit or result is not None:
                raise ResourceLimitError(
                    "HDF5 metadata exceeds the object, depth, or name-length limit"
                )
        return {
            "summary_type": "hdf5_structure_summary",
            "file_format": analyzer.file_extension.lstrip('.'),
            "file_size_bytes": analyzer.file_size,
            "object_count": len(objects),
            "objects": objects,
            "description": "HDF5 structure summary. Dataset payloads were not read.",
        }
    except ImportError:
        return summarize_opaque(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not inspect HDF5 file: {e}"}
