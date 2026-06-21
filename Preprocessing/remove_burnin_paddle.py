import argparse
import csv
import sys
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def load_image_as_grayscale_array(image_path):
    pil_image = Image.open(image_path).convert("L")
    return np.array(pil_image)

def build_brightness_candidate_mask(grayscale_image_array, brightness_threshold):
    return grayscale_image_array >= brightness_threshold

def extract_bounding_box_as_ints(paddle_quad_points):
    x_coords = [point[0] for point in paddle_quad_points]
    y_coords = [point[1] for point in paddle_quad_points]
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))

def bounding_box_overlaps_bright_region(x_min, y_min, x_max, y_max,brightness_candidate_mask):
    image_height, image_width = brightness_candidate_mask.shape
    clipped_y_min = max(0, y_min)
    clipped_y_max = min(image_height, y_max)
    clipped_x_min = max(0, x_min)
    clipped_x_max = min(image_width, x_max)

    region_mask = brightness_candidate_mask[clipped_y_min:clipped_y_max, clipped_x_min:clipped_x_max]
    return bool(region_mask.any())

def apply_black_box_censoring(grayscale_image_array, x_min, y_min, x_max, y_max, padding):
    image_height, image_width = grayscale_image_array.shape
    padded_x_min = max(0, x_min - padding)
    padded_y_min = max(0, y_min - padding)
    padded_x_max = min(image_width,  x_max + padding)
    padded_y_max = min(image_height, y_max + padding)

    censored_array = grayscale_image_array.copy()
    censored_array[padded_y_min:padded_y_max, padded_x_min:padded_x_max] = 0
    return censored_array

def collect_png_paths(input_dir, recursive):
    pattern = "**/*.png" if recursive else "*.png"
    return sorted(input_dir.glob(pattern))

def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR burn-in detetction and censor"
    )
    parser.add_argument("--input_dir",            "-i", required=True,  type=Path)
    parser.add_argument("--output_dir",           "-o", required=True,  type=Path)
    parser.add_argument("--brightness_threshold",       default=200,    type=int,
                        help="Pixels at or above threshold (0-255) are burn-in candidates (default: 200).")
    parser.add_argument("--confidence_threshold",       default=0.5,    type=float,
                        help="Minimum OCR confidence to consider a detection valid (default: 0.5).")
    parser.add_argument("--box_padding",                default=10,     type=int,
                        help="Extra pixels blacked out around each detected box (default: 10).")
    parser.add_argument("--report_csv",                 default="detections.csv", type=Path,
                        help="Path for the detection report CSV (default: detections.csv).")
    parser.add_argument("--recursive",            action="store_true")
    parser.add_argument("--no_detections_txt",          default="no_detections.txt", type=Path,
                        help="Path for the list of images where nothing was detected (default: no_detections.txt).")
    parser.add_argument("--ocr_border_padding",         default=50,     type=int,
                        help="Pixels of padding added around image before OCR to help catch corner annotations (default: 50).")
    parser.add_argument("--max_box_size",               default=150,    type=int,
                        help="Maximum allowed width and height of a detected bounding box in pixels (default: 150).")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: input directory not found - {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    png_paths = collect_png_paths(args.input_dir, args.recursive)
    if not png_paths:
        print("ERROR: no PNG files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(png_paths)} PNG files")
    print("Initialising PaddleOCR")

    ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False)
    report_rows = []
    images_with_nothing = []

    for image_path in tqdm(png_paths, unit="img"):
        grayscale_array = load_image_as_grayscale_array(image_path)
        brightness_mask = build_brightness_candidate_mask(grayscale_array, args.brightness_threshold)

        rgb_array = cv2.cvtColor(grayscale_array, cv2.COLOR_GRAY2RGB)
        ocr_border_padding = args.ocr_border_padding
        padded_rgb_array = cv2.copyMakeBorder(
            rgb_array,
            ocr_border_padding, ocr_border_padding,
            ocr_border_padding, ocr_border_padding,
            cv2.BORDER_CONSTANT, value=0
        )


        ocr_result = ocr_engine.predict(padded_rgb_array)
        ocr_result_object = ocr_result[0] if ocr_result else None

        censored_array = grayscale_array.copy()
        number_of_detections_in_image = 0

        if not ocr_result_object:
            Image.fromarray(censored_array, mode="L").save(str(args.output_dir / image_path.name))
            continue

        detected_texts    = ocr_result_object["rec_texts"]
        detected_scores   = ocr_result_object["rec_scores"]
        detected_polygons = ocr_result_object["rec_polys"]

        for detected_text, confidence_score, quad_points in zip(detected_texts, detected_scores, detected_polygons):

            if confidence_score < args.confidence_threshold:
                continue

            normalized_detected_text = detected_text.strip().upper()
            if normalized_detected_text not in {"L", "R", "P"}:
                continue

            x_min, y_min, x_max, y_max = extract_bounding_box_as_ints(quad_points)

            x_min -= ocr_border_padding
            y_min -= ocr_border_padding
            x_max -= ocr_border_padding
            y_max -= ocr_border_padding

            box_width  = x_max - x_min
            box_height = y_max - y_min
            if box_width > args.max_box_size or box_height > args.max_box_size:
                continue

            if not bounding_box_overlaps_bright_region(x_min, y_min, x_max, y_max, brightness_mask):
                continue

            censored_array = apply_black_box_censoring(censored_array, x_min, y_min, x_max, y_max, args.box_padding)
            number_of_detections_in_image += 1

            report_rows.append({
                "image":      image_path.name,
                "text":       detected_text,
                "confidence": round(confidence_score, 4),
                "x_min":      x_min,
                "y_min":      y_min,
                "x_max":      x_max,
                "y_max":      y_max,
            })
        if number_of_detections_in_image == 0:
            images_with_nothing.append(image_path.name)

        output_image_path = args.output_dir / image_path.name
        Image.fromarray(censored_array, mode="L").save(str(output_image_path))

        if number_of_detections_in_image > 0:
            tqdm.write(f"   {image_path.name}: {number_of_detections_in_image} regions censored")

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_csv, "w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = ["image", "text", "confidence", "x_min", "y_min", "x_max", "y_max"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    total_images_with_detections = len({row["image"] for row in report_rows})
    print(f"\n Done: {total_images_with_detections}/{len(png_paths)} images had detections.")
    print(f"Report saved into {args.report_csv}")

    args.no_detections_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(args.no_detections_txt, "w") as no_detections_file:
        no_detections_file.write("\n".join(images_with_nothing))
    print(f"Images with no detections ({len(images_with_nothing)}) saved into file {args.no_detections_txt}")


if __name__ == "__main__":
    main()
