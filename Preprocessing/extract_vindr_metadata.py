import argparse
import sys
import pandas as pd
import pydicom
from pathlib import Path

def collect_dicom_paths(dicom_dir, recursive):
    pattern = "**/*.dicom" if recursive else "*.dicom"
    dicom_paths = sorted(dicom_dir.glob(pattern))
 
    if not dicom_paths:
        glob_pattern = "**/*" if recursive else "*"
        extensionless_dicoms = []
        for candidate_path in dicom_dir.glob(glob_pattern):
            if candidate_path.is_file() and not candidate_path.suffix:
                try:
                    pydicom.dcmread(str(candidate_path), stop_before_pixels=True)
                    extensionless_dicoms.append(candidate_path)
                except Exception:
                    pass
        dicom_paths = sorted(extensionless_dicoms)
 
    return dicom_paths

def extract_pixel_spacing(dataset):
    pixel_spacing_element = dataset.get((0x0028, 0x0030))
    if pixel_spacing_element is not None:
        spacing_values = pixel_spacing_element.value
        row_spacing = str(spacing_values[0]) if len(spacing_values) > 0 else ""
        col_spacing = str(spacing_values[1]) if len(spacing_values) > 1 else ""
        return row_spacing, col_spacing
    return "", ""

def read_dicom_tag(dataset, tag, default=""):
    element = dataset.get(tag)
    if element is None:
        return default
    value = element.value
    if hasattr(value, "__iter__") and not isinstance(value, str):
        return str(list(value))
    return value

def extract_metadata_from_dicom(dicom_path):
    dataset = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    pixel_spacing_x, pixel_spacing_y = extract_pixel_spacing(dataset)
    row = {
        "Image Index":                      dicom_path.stem,
        "Patient Age":                      read_dicom_tag(dataset, (0x0010, 0x1010)),
        "Patient Sex":                      read_dicom_tag(dataset, (0x0010, 0x0040)),
        "OriginalImage[Width]":             read_dicom_tag(dataset, (0x0028, 0x0011)),
        "OriginalImage[Height]":            read_dicom_tag(dataset, (0x0028, 0x0010)),
        "OriginalImagePixelSpacing[x]":     pixel_spacing_x,
        "OriginalImagePixelSpacing[y]":     pixel_spacing_y,
        "Patient Size":                     read_dicom_tag(dataset, (0x0010, 0x1020)),
        "Patient Weight":                   read_dicom_tag(dataset, (0x0010, 0x1030)),
        "Pixel Aspect Ratio":               read_dicom_tag(dataset, (0x0028, 0x0034)),
        "Bits Allocated":                   read_dicom_tag(dataset, (0x0028, 0x0100)),
        "Bits Stored":                      read_dicom_tag(dataset, (0x0028, 0x0101)),
        "High Bit":                         read_dicom_tag(dataset, (0x0028, 0x0102)),
        "Pixel Representation":             read_dicom_tag(dataset, (0x0028, 0x0103)),
        "Smallest Image Pixel Value":       read_dicom_tag(dataset, (0x0028, 0x0106)),
        "Largest Image Pixel Value":        read_dicom_tag(dataset, (0x0028, 0x0107)),
        "Window Center":                    read_dicom_tag(dataset, (0x0028, 0x1050)),
        "Window Width":                     read_dicom_tag(dataset, (0x0028, 0x1051)),
        "Rescale Intercept":                read_dicom_tag(dataset, (0x0028, 0x1052)),
        "Rescale Slope":                    read_dicom_tag(dataset, (0x0028, 0x1053)),
        "Photometric Interpretation":       read_dicom_tag(dataset, (0x0028, 0x0004)),
        "Samples per Pixel":                read_dicom_tag(dataset, (0x0028, 0x0002)),
        "Number of Frames":                 read_dicom_tag(dataset, (0x0028, 0x0008)),
        "Lossy Image Compression":          read_dicom_tag(dataset, (0x0028, 0x2110)),
        "Lossy Image Compression Method":   read_dicom_tag(dataset, (0x0028, 0x2114)),
        "Image Compression Ratio":          read_dicom_tag(dataset, (0x0028, 0x2112)),
    }
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir",  "-d", required=True, type=Path)
    parser.add_argument("--output_csv", "-o", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true", help="bool: search dir recursively.")
    args = parser.parse_args()

    if not args.dicom_dir.exists():
        print(f"ERROR: DICOM directory not found: {args.dicom_dir}", file=sys.stderr)
        sys.exit(1)
 
    dicom_paths = collect_dicom_paths(args.dicom_dir, args.recursive)
    if not dicom_paths:
        print("ERROR: No DICOM files found.", file=sys.stderr)
        sys.exit(1)
        print(f"Found {len(dicom_paths)} DICOM file(s). Extracting metadata...")
 
    all_rows = []
    failed_files = []
 
    for dicom_path in dicom_paths:
        try:
            metadata_row = extract_metadata_from_dicom(dicom_path)
            all_rows.append(metadata_row)
        except Exception as error:
            print(f"WARNING: could not read {dicom_path.name}: {error}", file=sys.stderr)
            failed_files.append(dicom_path.name)
 
    metadata_dataframe = pd.DataFrame(all_rows)
 
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_dataframe.to_csv(args.output_csv, index=False)
 
    print(f"Saved {len(all_rows)} rows → {args.output_csv}")
    if failed_files:
        print(f"Failed ({len(failed_files)}): {', '.join(failed_files)}", file=sys.stderr)
 
 
if __name__ == "__main__":
    main()