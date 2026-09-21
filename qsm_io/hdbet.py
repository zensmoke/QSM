# qsm_io/hdbet.py
from pathlib import Path
import subprocess

def run_hdbet(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    hdbet_out = output_dir / "brain.nii.gz"

    cmd = [
        "hd-bet",
        "-i", str(input_path),
        "-o", str(hdbet_out),
        "--save_bet_mask",
        "--no_bet_image",
        "-device", "cpu",
    ]
    print(f"[HD-BET] Запуск: {' '.join(cmd)}")

    import subprocess
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        # HD-BET печатает прогресс в stderr — покажем только хвост
        tail = "\n".join(result.stderr.strip().splitlines()[-10:])
        raise RuntimeError(f"HD-BET упал (code {result.returncode}):\n{tail}")

    mask_path = output_dir / "brain_mask.nii.gz"
    if not mask_path.exists():
        cands = list(output_dir.glob("*mask*.nii.gz"))
        if not cands:
            raise FileNotFoundError("HD-BET не создал маску")
        mask_path = cands[0]

    print(f"[HD-BET] Маска: {mask_path}")
    return mask_path