import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from ensemble_boxes import weighted_boxes_fusion


MINIMUM_RADIOLOGIST_AGREEMENT_SCORE = 0.33

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",       "-i", required=True,  type=Path)
    parser.add_argument("--metadata_csv",    "-m", required=True,  type=Path)
    parser.add_argument("--output_csv",      "-o", required=True,  type=Path)
    parser.add_argument("--iou_threshold",         default=0.4,    type=float)
    args = parser.parse_args()

    if not args.input_csv.exists():
        print(f"ERROR: {args.input_csv} not found", file=sys.stderr)
        sys.exit(1)
    if not args.metadata_csv.exists():
        print(f"ERROR: {args.metadata_csv} not found", file=sys.stderr)
        sys.exit(1)

    annotation_dataframe = pd.read_csv(args.input_csv)
    metadata_dataframe    = pd.read_csv(args.metadata_csv)

    image_dimensions = metadata_dataframe.set_index("image_id")[
        ["OriginalImage[Width]", "OriginalImage[Height]"]
    ].to_dict(orient="index")

    number_of_radiologists_per_image = (
        annotation_dataframe.groupby("image_id")["rad_id"].nunique()
    )

    all_output_rows = []

    for image_id, annotations in annotation_dataframe.groupby("image_id"):
        if image_id not in image_dimensions:
            print(f"WARNING: no metadata for {image_id}, skipping", file=sys.stderr)
            continue

        image_width  = image_dimensions[image_id]["OriginalImage[Width]"]
        image_height = image_dimensions[image_id]["OriginalImage[Height]"]
        total_radiologists_for_image = number_of_radiologists_per_image[image_id]

        for class_name, class_annotations in annotations.groupby("class_name"):
            normalized_boxes = []
            confidence_scores = []

            for _, annotation_row in class_annotations.iterrows():
                normalized_x_min = annotation_row["x_min"] / image_width
                normalized_y_min = annotation_row["y_min"] / image_height
                normalized_x_max = annotation_row["x_max"] / image_width
                normalized_y_max = annotation_row["y_max"] / image_height

                normalized_x_min = float(np.clip(normalized_x_min, 0.0, 1.0))
                normalized_y_min = float(np.clip(normalized_y_min, 0.0, 1.0))
                normalized_x_max = float(np.clip(normalized_x_max, 0.0, 1.0))
                normalized_y_max = float(np.clip(normalized_y_max, 0.0, 1.0))

                normalized_boxes.append(
                    [normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max]
                )
                confidence_scores.append(1.0)

            fused_boxes, fused_scores, _ = weighted_boxes_fusion(
                [normalized_boxes],
                [confidence_scores],
                [np.zeros(len(normalized_boxes), dtype=int).tolist()],
                iou_thr=args.iou_threshold,
                skip_box_thr=0.0,
            )

            for fused_box, fused_score in zip(fused_boxes, fused_scores):
                agreement_score = fused_score / total_radiologists_for_image

                if agreement_score < MINIMUM_RADIOLOGIST_AGREEMENT_SCORE:
                    continue

                all_output_rows.append({
                    "image_id":              image_id,
                    "class_name":            class_name,
                    "x_min":                 round(fused_box[0] * image_width,  4),
                    "y_min":                 round(fused_box[1] * image_height, 4),
                    "x_max":                 round(fused_box[2] * image_width,  4),
                    "y_max":                 round(fused_box[3] * image_height, 4),
                    "agreement_score":       round(agreement_score,  4),
                })

    output_dataframe = pd.DataFrame(all_output_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(args.output_csv, index=False)
    print(f"Saved {len(output_dataframe)} fused boxes in {args.output_csv}")


if __name__ == "__main__":
    main()