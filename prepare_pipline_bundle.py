import shutil
from pathlib import Path

# Paths that make up the runtime pipeline bundle
ITEMS_TO_COPY = [
    "config.json",
    "config_gazebo.json",
    "requirements.txt",
    "mobile_camera_pointcloud.py",
    "depth_to_pointcloud.py",
    "process_sequence.py",
    "options.py",
    "utils.py",
    "lib",
    "cam_calibration",
    "networks",
    "weights/RTMonoDepth",
    "gazebo_simulation",
    "ros_bridge",
]

DEST_FOLDER = Path("pipline")


def copy_item(src_root: Path, relative_path: str) -> None:
    src_path = src_root / relative_path
    dest_path = DEST_FOLDER / relative_path
    if not src_path.exists():
        print(f"[WARN] Skipping missing item: {relative_path}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_dir():
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)
    else:
        shutil.copy2(src_path, dest_path)

    print(f"[OK] Copied {relative_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    if DEST_FOLDER.exists():
        print("[INFO] Existing 'pipline' folder found; removing to rebuild bundle...")
        shutil.rmtree(DEST_FOLDER)

    DEST_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created bundle root: {DEST_FOLDER}")

    for item in ITEMS_TO_COPY:
        copy_item(repo_root, item)

    print("[DONE] Pipeline bundle refreshed in 'pipline/'")


if __name__ == "__main__":
    main()
