import zipfile
import tarfile
from typing import Dict, Any

from .utility import summarize_opaque

def summarize_archive(analyzer) -> Dict[str, Any]:
    try:
        file_list = []
        if analyzer.file_extension == '.zip' and zipfile.is_zipfile(analyzer.file_path):
            with zipfile.ZipFile(analyzer.file_path, 'r') as zf:
                file_list = zf.namelist()
        elif tarfile.is_tarfile(analyzer.file_path):
            with tarfile.open(analyzer.file_path, 'r:*') as tf:
                file_list = tf.getnames()
        else:
            return summarize_opaque(analyzer)

        summary = {
            "summary_type": "archive_contents",
            "file_format": analyzer.file_extension.lstrip('.'),
            "file_count": len(file_list),
            "file_list": file_list[:analyzer.MAX_ARCHIVE_LIST_FILES]
        }
        if len(file_list) > analyzer.MAX_ARCHIVE_LIST_FILES:
            summary['file_list'].append(f"... ({len(file_list) - analyzer.MAX_ARCHIVE_LIST_FILES} more files)")
        return summary
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not inspect archive: {e}"}
