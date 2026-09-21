from pathlib import Path
import numpy as np
import nibabel as nib

def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine, img.header

def save_nifti(data: np.ndarray, affine: np.ndarray,
               header: nib.Nifti1Header | None, path: str | None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is not None:
        header.set_data_dtype(np.float32)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(img, str(path))
    print(f"[save] {path}")

def get_zooms(header: nib.Nifti1Header) -> tuple[float, float, float]:
    return tuple(float(z) for z in header.get_zooms()[:3])
