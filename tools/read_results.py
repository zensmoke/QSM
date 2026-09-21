"""
Чтение XLSX с результатами KCL.
"""
from pathlib import Path
import pandas as pd


def show_results(path: Path, label: str):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'=' * 60}")
    try:
        df = pd.read_excel(path)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  Ошибка: {e}")


if __name__ == "__main__":
    KCL = Path("/KCL")

    # Оригинальные результаты KCL
    show_results(KCL / "results/NM_CNR_SNVTA_Results.xlsx",
                  "ОРИГИНАЛ KCL")

    # Наши результаты
    show_results(KCL / "results_test/NM_CNR_SNVTA_Results.xlsx",
                  "НАШ ЗАПУСК")

    # Сравнение
    print(f"\n{'=' * 60}")
    print("  СРАВНЕНИЕ")
    print(f"{'=' * 60}")

    try:
        df_orig = pd.read_excel(KCL / "results/NM_CNR_SNVTA_Results.xlsx")
        df_ours = pd.read_excel(KCL / "results_test/NM_CNR_SNVTA_Results.xlsx")

        for sid in df_orig["Subject ID"]:
            o = df_orig[df_orig["Subject ID"] == sid].iloc[0]
            m = df_ours[df_ours["Subject ID"] == sid]
            if m.empty:
                print(f"  {sid}: не найден в нашем запуске")
                continue
            m = m.iloc[0]

            d_mean = abs(o["Mean"] - m["Mean"])
            d_std = abs(o["Standard Deviation"] - m["Standard Deviation"])

            print(f"  {sid}:")
            print(f"    KCL:  Mean = {o['Mean']:.6f}, SD = {o['Standard Deviation']:.6f}")
            print(f"    Наши: Mean = {m['Mean']:.6f}, SD = {m['Standard Deviation']:.6f}")
            print(f"    Δ Mean = {d_mean:.6f}")
            print(f"    Δ SD   = {d_std:.6f}")

            if d_mean < 1e-4 and d_std < 1e-4:
                print(f"    ✅ ПОЛНОЕ СОВПАДЕНИЕ")
            elif d_mean < 1e-3:
                print(f"    ✅ Отличное совпадение")
            else:
                print(f"    ⚠️  Есть расхождение")
    except Exception as e:
        print(f"  Ошибка: {e}")