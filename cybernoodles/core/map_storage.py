import os
import zipfile


def _should_keep_map_entry(name):
    """Keep only beatmap data files needed for parsing and training."""
    base = os.path.basename(name).lower()
    if not base:
        return False
    return base.endswith(".dat")


def slim_map_archive(zip_path):
    """Rewrite a BeatSaver zip so only .dat files remain.

    This keeps every beatmap data file (Info.dat, BPMInfo.dat, difficulties,
    lightshow .dat files, etc.) while dropping audio, covers, and extras.
    """
    if not os.path.exists(zip_path):
        return False, 0, 0

    before_bytes = os.path.getsize(zip_path)
    temp_path = f"{zip_path}.tmp"

    try:
        with zipfile.ZipFile(zip_path, "r") as src:
            file_infos = [info for info in src.infolist() if not info.is_dir()]
            keep_infos = [info for info in file_infos if _should_keep_map_entry(info.filename)]
            discard_infos = [info for info in file_infos if not _should_keep_map_entry(info.filename)]

            # Already slim, or the archive is malformed and contains no usable data.
            if not discard_infos or not keep_infos:
                return False, before_bytes, before_bytes

            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as dst:
                for info in keep_infos:
                    dst.writestr(info.filename, src.read(info.filename))

        os.replace(temp_path, zip_path)
        after_bytes = os.path.getsize(zip_path)
        return True, before_bytes, after_bytes
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def slim_map_cache(maps_dir):
    """Slim every cached BeatSaver archive in-place."""
    total_before = 0
    total_after = 0
    trimmed = 0

    if not os.path.isdir(maps_dir):
        return {
            "trimmed": 0,
            "before_bytes": 0,
            "after_bytes": 0,
            "saved_bytes": 0,
        }

    for name in os.listdir(maps_dir):
        if not name.lower().endswith(".zip"):
            continue

        zip_path = os.path.join(maps_dir, name)
        changed, before_bytes, after_bytes = slim_map_archive(zip_path)
        total_before += before_bytes
        total_after += after_bytes
        if changed:
            trimmed += 1

    return {
        "trimmed": trimmed,
        "before_bytes": total_before,
        "after_bytes": total_after,
        "saved_bytes": max(0, total_before - total_after),
    }
