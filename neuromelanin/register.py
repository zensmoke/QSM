"""
ЭТАП 1: ANTs регистрация (без nibabel).

Логика:
  1. T1 → MNI (SyN/Affine) с авто-fallback на SyN
  2. NM (echo-1) → T1 (Rigid)
  3. Атлас MNI → T1 → NM — пошагово
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import ants


def log(msg, level="info"):
    prefix = {"info": "  ", "step": "\n▶", "ok": "  ✅", "err": "  ❌",
              "warn": "  ⚠️ "}.get(level, "  ")
    print(f"{prefix} {msg}", flush=True)


def safe_path(src: Path, tmp: Path) -> Path:
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Нет файла: {src}")
    dst = tmp / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-t1", type=Path, required=True)
    p.add_argument("--nm-mri", type=Path, required=True)
    p.add_argument("--mni-t1", type=Path, required=True)
    p.add_argument("--atlas-mni", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--type", default="SyN")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    log(f"ANTs {ants.__version__}, threads={args.threads}", "step")

    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="ants_reg_"))
    log(f"tmp: {tmp}", "step")

    try:
        log("Копирую файлы в /tmp...")
        t1_safe = safe_path(args.subject_t1, tmp)
        nm_safe = safe_path(args.nm_mri, tmp)
        mni_safe = safe_path(args.mni_t1, tmp)
        atlas_safe = safe_path(args.atlas_mni, tmp)

        log("Загружаю через ANTs...")
        t1_img = ants.image_read(str(t1_safe))
        nm_img = ants.image_read(str(nm_safe))
        mni_img = ants.image_read(str(mni_safe))
        atlas_img = ants.image_read(str(atlas_safe))

        log(f"  T1:    shape={t1_img.shape}, spacing="
            f"{tuple(f'{s:.2f}' for s in t1_img.spacing)}")
        log(f"  NM:    shape={nm_img.shape}, spacing="
            f"{tuple(f'{s:.2f}' for s in nm_img.spacing)}")
        log(f"  MNI:   shape={mni_img.shape}")

        # ========== ШАГ 1: T1 → MNI ==========
        log(f"\n[1/3] Регистрация T1 → MNI ({args.type})...")

        # Минимальное число вокселей атласа в T1 для успешной регистрации
        # Affine на T1_sag даёт ~2300 — это плохо, нужно 10000+
        MIN_ATLAS_VOXELS = 5000

        def do_reg(reg_type):
            log(f"  Пробую {reg_type}...")
            return ants.registration(
                fixed=mni_img, moving=t1_img,
                type_of_transform=reg_type, verbose=False,
            )

        def count_atlas_in_t1(reg):
            atlas_t = ants.apply_transforms(
                fixed=t1_img, moving=atlas_img,
                transformlist=reg["invtransforms"],
                interpolator="multiLabel",
            )
            arr = atlas_t.numpy()
            n1 = int((arr == 1).sum())
            n2 = int((arr == 2).sum())
            return n1 + n2, n1, n2

        reg_t1_mni = do_reg(args.type)
        n_total, n_crus, n_sn = count_atlas_in_t1(reg_t1_mni)
        log(f"    После {args.type}: {n_total} вокселей "
            f"(Crus={n_crus}, SN={n_sn})")

        used_type = args.type

        # Авто-fallback на SyN, если меньше MIN_ATLAS_VOXELS
        if n_total < MIN_ATLAS_VOXELS and args.type != "SyN":
            log(f"  ⚠ {args.type} дал {n_total} вокселей "
                f"(< {MIN_ATLAS_VOXELS}) — fallback на SyN", "warn")
            reg_t1_mni = do_reg("SyN")
            n_total, n_crus, n_sn = count_atlas_in_t1(reg_t1_mni)
            log(f"    После SyN: {n_total} вокселей "
                f"(Crus={n_crus}, SN={n_sn})")
            used_type = "SyN"

        if n_total < MIN_ATLAS_VOXELS:
            raise RuntimeError(
                f"Регистрация T1 → MNI провалилась: {n_total} вокселей "
                f"(< {MIN_ATLAS_VOXELS}). Попробуйте T1_tra вместо T1_sag."
            )

        log(f"  ✅ T1 → MNI ({used_type})")

        # ========== ШАГ 2: NM → T1 ==========
        log("\n[2/3] Регистрация NM → T1 (Rigid)...")
        reg_nm_t1 = ants.registration(
            fixed=t1_img, moving=nm_img,
            type_of_transform="Rigid", verbose=False,
        )
        log(f"  ✅ NM → T1")

        # ========== ШАГ 3: Атлас MNI → T1 → NM ==========
        log("\n[3/3] Перенос атласа MNI → NM (пошагово)...")

        log("  [3.1] Атлас MNI → T1...")
        atlas_in_t1 = ants.apply_transforms(
            fixed=t1_img, moving=atlas_img,
            transformlist=reg_t1_mni["invtransforms"],
            interpolator="multiLabel",
        )
        n1 = int((atlas_in_t1.numpy() > 0).sum())
        log(f"       ненулевых: {n1}")
        if n1 < 500:
            raise RuntimeError(f"3.1 провалился: {n1} вокселей")

        log("  [3.2] Атлас T1 → NM...")
        atlas_native_img = ants.apply_transforms(
            fixed=nm_img, moving=atlas_in_t1,
            transformlist=reg_nm_t1["invtransforms"],
            interpolator="multiLabel",
        )
        n2 = int((atlas_native_img.numpy() > 0).sum())
        log(f"       ненулевых: {n2}")
        if n2 < 500:
            raise RuntimeError(f"3.2 провалился: {n2} вокселей")

        log(f"  ✅ Атлас в пространстве NM: {n2} вокселей")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(atlas_native_img, str(args.output))

        elapsed = time.time() - t0
        meta = {
            "type_t1_mni": used_type,
            "type_nm_t1": "Rigid",
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
            "antspyx_version": ants.__version__,
            "n_nonzero_atlas_nm": n2,
        }
        args.meta.parent.mkdir(parents=True, exist_ok=True)
        args.meta.write_text(json.dumps(meta, indent=2))

        log(f"\n✅ Атлас: {args.output}")
        log(f"⏱ Время: {elapsed:.0f}s")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()