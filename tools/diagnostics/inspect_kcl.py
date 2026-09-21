# inspect_kcl.py
from pathlib import Path
import nibabel as nib
import numpy as np

# Настройте путь к папке KCL
kcl_dir = Path("/KCL")  # или как называется папка

if not kcl_dir.exists():
    print(f"Папка не найдена: {kcl_dir}")
    # Поищем папки с "kcl" или "neuromelanin" в имени
    project_root = Path("/")
    print("\nВозможные кандидаты:")
    for p in project_root.iterdir():
        if p.is_dir() and ("kcl" in p.name.lower() or "nm" in p.name.lower()
                            or "atlas" in p.name.lower()):
            print(f"   {p}")
    exit(1)

print(f"Папка: {kcl_dir.resolve()}\n")

def show_dir(d, indent=0):
    """Рекурсивно показывает содержимое."""
    prefix = "  " * indent
    items = sorted(d.iterdir())
    for item in items:
        if item.is_dir():
            print(f"{prefix}[DIR] {item.name}/")
            show_dir(item, indent + 1)
        else:
            size_kb = item.stat().st_size / 1024
            extra = ""
            if item.suffix in (".nii", ".gz") or ".nii" in item.name:
                try:
                    img = nib.load(str(item))
                    extra = f"  shape={img.shape}"
                except Exception:
                    extra = "  (не NIfTI)"
            print(f"{prefix}[FILE] {item.name}  "
                  f"({size_kb:.1f} KB){extra}")

show_dir(kcl_dir)