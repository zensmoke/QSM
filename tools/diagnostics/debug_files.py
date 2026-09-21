# debug_files.py
from pathlib import Path

data_dir = Path('/Users/nikitamyasov/3782185 ласточка/data/')

print(f"Существует: {data_dir.exists()}")
print(f"Это папка:  {data_dir.is_dir()}")
print(f"Абсолютный путь: {data_dir.resolve()}")
print()

if data_dir.exists():
    files = sorted(data_dir.iterdir())
    print(f"Найдено объектов: {len(files)}\n")
    for f in files:
        if f.is_file():
            print(f"  [FILE] {f.name}   (size: {f.stat().st_size} bytes)")
        else:
            print(f"  [DIR]  {f.name}")