from pathlib import Path


DATA_DIR = Path("data/LOL-v2-real")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

if not DATA_DIR.exists():
    print("Dataset folder not found!")
    print("Expected location:", DATA_DIR.resolve())
else:
    print("Dataset location:", DATA_DIR.resolve())
    for folder in sorted(DATA_DIR.iterdir()):
        if folder.is_dir() and not folder.name.startswith("."):
            print(f"\n{folder.name}/")
            for subfolder in sorted(folder.iterdir()):
                if subfolder.is_dir():
                    images = [
                        path
                        for path in subfolder.rglob("*")
                        if path.is_file()
                        and path.suffix.lower() in IMAGE_EXTENSIONS
                    ]
                    print(f"  {subfolder.name}/: {len(images)} images")
