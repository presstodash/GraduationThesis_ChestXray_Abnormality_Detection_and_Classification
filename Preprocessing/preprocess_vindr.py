import argparse
import os
import sys
import logging
import numpy as np
import pydicom
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def process_dicom( dicom_path: Path, output_dir: Path, bit_depth: int = 8, clip_percentile: float = 1.0, do_clip: bool = True, ) -> tuple[str, bool, str]:
    try:
        dataset = pydicom.dcmread(str(dicom_path))
        pixel_array = dataset.pixel_array.astype(np.float64)

        pixel_array = apply_modality_lut(pixel_array, dataset)

        pixel_array = apply_voi_lut(pixel_array, dataset, prefer_lut=True)

        photometric = getattr(dataset, "PhotometricInterpretation", "").strip().upper()
        if photometric == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array

        if do_clip and clip_percentile > 0.0:
            p_low = np.percentile(pixel_array, clip_percentile)
            p_high = np.percentile(pixel_array, 100.0 - clip_percentile)
            pixel_array = np.clip(pixel_array, p_low, p_high)

        v_min = pixel_array.min()
        v_max = pixel_array.max()

        if v_max > v_min:
            pixel_array = (pixel_array - v_min) / (v_max - v_min)
        else:
            pixel_array = np.full_like(pixel_array, 0.5)

        if bit_depth == 8:
            pixel_array = (pixel_array * 255.0).round().astype(np.uint8)
            pil_mode = "L"
        elif bit_depth == 16:
            pixel_array = (pixel_array * 65535.0).round().astype(np.uint16)
            pil_mode = "I;16"
        else:
            raise ValueError(f"Unsupported bit_depth={bit_depth}. Choose 8 or 16.")

        img = Image.fromarray(pixel_array, mode="L" if bit_depth == 8 else "I")
        img = img.resize((1024, 1024), Image.LANCZOS)

        out_name = dicom_path.stem + ".png"
        out_path = output_dir / out_name
        img.save(str(out_path))

        return (dicom_path.name, True, "")

    except Exception as exc:
        return (dicom_path.name, False, str(exc))
    
def collect_dicoms(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.dicom" if recursive else "*.dicom"
    files = sorted(input_dir.glob(pattern))

    if not files:
        log.warning(
            "No *.dicom files found.  Scanning for extension-less DICOMs..."
        )
        candidates = []
        glob_pat = "**/*" if recursive else "*"
        for p in input_dir.glob(glob_pat):
            if p.is_file() and not p.suffix:
                try:
                    pydicom.dcmread(str(p), stop_before_pixels=True)
                    candidates.append(p)
                except Exception:
                    pass
        files = sorted(candidates)

    return files

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", required=True, type=Path,
        help="Directory containing DICOM files."
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, type=Path,
        help="Directory where PNG files will be saved."
    )
    parser.add_argument(
        "--bit_depth", type=int, default=8, choices=[8, 16],
        help="Output bit depth per channel (default: 8)."
    )
    parser.add_argument(
        "--clip_percentile", type=float, default=1.0,
        help=(
            "Percentile for symmetric robust clipping, e.g. 1.0 clips the "
            "bottom 1%% and top 1%% of intensities (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--no_clip", action="store_true",
        help="Disable robust clipping entirely."
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes (default: cpu count)."
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search input_dir recursively for DICOM files."
    )
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    if not args.input_dir.exists():
        log.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dicom_files = collect_dicoms(args.input_dir, args.recursive)
    if not dicom_files:
        log.error("No DICOM files found. Check --input_dir and --recursive flag.")
        sys.exit(1)

    log.info(f"Found {len(dicom_files)} DICOM file(s).")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Bit depth       : {args.bit_depth}")
    log.info(f"Robust clipping : {'disabled' if args.no_clip else f'{args.clip_percentile}%'}")
    log.info(f"Workers         : {args.workers}")

    worker = partial(
        process_dicom,
        output_dir=args.output_dir,
        bit_depth=args.bit_depth,
        clip_percentile=args.clip_percentile,
        do_clip=not args.no_clip,
    )

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, path): path for path in dicom_files}
        with tqdm(total=len(dicom_files), unit="img") as pbar:
            for future in as_completed(futures):
                fname, ok, err = future.result()
                if not ok:
                    log.warning(f"FAILED {fname}: {err}")
                    failed.append((fname, err))
                pbar.update(1)

    n_ok = len(dicom_files) - len(failed)
    log.info(f"Done. {n_ok}/{len(dicom_files)} images converted successfully.")
    if failed:
        log.warning(f"{len(failed)} file(s) failed:")
        for fname, err in failed:
            log.warning(f"  {fname}: {err}")
        sys.exit(2)


if __name__ == "__main__":
    main()