"""
APTIVA AI — ZIP Loader
======================
Extracts and validates PDF and DOCX files from uploaded ZIP archives.
Handles nested folders, skips hidden files, and preserves full relative paths
to prevent filename collisions.
"""

import io
import zipfile
from typing import List, Tuple

def extract_resumes_from_zip(zip_bytes: bytes) -> Tuple[List[Tuple[bytes, str, str]], List[Tuple[str, str]]]:
    """
    Extracts PDF and DOCX resumes from a ZIP archive.
    
    Parameters
    ----------
    zip_bytes : bytes
        The raw bytes of the uploaded ZIP file.
        
    Returns
    -------
    Tuple containing:
        - List of successful extractions: [(file_bytes, filename, relative_path)]
        - List of skipped/failed files: [(relative_path, reason)]
    """
    successful_files = []
    skipped_files = []
    
    MAX_FILES = 100
    MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB limit for uncompressed data
    total_size_extracted = 0
    file_count = 0
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for zip_info in z.infolist():
                if zip_info.is_dir():
                    continue
                    
                relative_path = zip_info.filename
                filename = relative_path.split("/")[-1]
                
                # Skip hidden files and macOS metadata
                if filename.startswith(".") or "__MACOSX" in relative_path:
                    continue
                    
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext not in ["pdf", "docx"]:
                    skipped_files.append((relative_path, "Unsupported format"))
                    continue
                
                if zip_info.file_size > 5 * 1024 * 1024:
                    skipped_files.append((relative_path, "File exceeds 5MB limit"))
                    continue
                    
                file_count += 1
                if file_count > MAX_FILES:
                    skipped_files.append((relative_path, f"Archive exceeds maximum of {MAX_FILES} files"))
                    continue
                
                total_size_extracted += zip_info.file_size
                if total_size_extracted > MAX_TOTAL_SIZE:
                    skipped_files.append((relative_path, "Archive exceeds total uncompressed size limit of 100MB"))
                    break
                
                try:
                    file_bytes = z.read(zip_info.filename)
                    successful_files.append((file_bytes, filename, relative_path))
                except Exception as e:
                    skipped_files.append((relative_path, f"Extraction failed: {str(e)}"))
                    
    except zipfile.BadZipFile:
        skipped_files.append(("ZIP Archive", "Corrupted or invalid ZIP file"))
    except RuntimeError as e:
        if "encrypted" in str(e).lower() or "password" in str(e).lower():
            skipped_files.append(("ZIP Archive", "Password-protected archive"))
        else:
            skipped_files.append(("ZIP Archive", f"Extraction error: {str(e)}"))
    except Exception as e:
        skipped_files.append(("ZIP Archive", f"Unexpected error: {str(e)}"))
        
    return successful_files, skipped_files
