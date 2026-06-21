import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


NO_FINDING_LABEL = "No finding"

REQUIRED_ANNOTATION_COLUMNS = [
    "image_id",
    "class_name",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]

REQUIRED_METADATA_COLUMNS = [
    "image_id",
    "OriginalImage[Width]",
    "OriginalImage[Height]",
]

COORDINATE_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]


def validate_columns(dataframe: pd.DataFrame, required_columns: list[str], source_name: str):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{source_name} is missing columns: {missing_columns}")


def load_class_config(class_config_json_path: Path) -> dict:
    if not class_config_json_path.exists():
        raise FileNotFoundError(f"class_config_json not found: {class_config_json_path}")

    with open(class_config_json_path, "r", encoding="utf-8") as json_file:
        raw_config = json.load(json_file)

    class_config = raw_config.get("classes")
    if not isinstance(class_config, dict) or not class_config:
        raise ValueError("class_config_json must contain a non-empty 'classes' object")

    for class_name, cfg in class_config.items():
        if "class_id" not in cfg:
            raise ValueError(f"Missing class_id for class '{class_name}' in class_config_json")

    return class_config


def load_inputs(input_csv_path: Path, metadata_csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations_dataframe = pd.read_csv(input_csv_path)
    metadata_dataframe = pd.read_csv(metadata_csv_path)

    validate_columns(annotations_dataframe, REQUIRED_ANNOTATION_COLUMNS, "input_csv")
    validate_columns(metadata_dataframe, REQUIRED_METADATA_COLUMNS, "metadata_csv")

    annotations_dataframe["image_id"] = annotations_dataframe["image_id"].astype(str)
    metadata_dataframe["image_id"] = metadata_dataframe["image_id"].astype(str)

    return annotations_dataframe, metadata_dataframe


def validate_metadata(metadata_dataframe: pd.DataFrame):
    metadata_dataframe["OriginalImage[Width]"] = pd.to_numeric(
        metadata_dataframe["OriginalImage[Width]"], errors="coerce"
    )
    metadata_dataframe["OriginalImage[Height]"] = pd.to_numeric(
        metadata_dataframe["OriginalImage[Height]"], errors="coerce"
    )

    invalid = metadata_dataframe[
        metadata_dataframe["OriginalImage[Width]"].isna()
        | metadata_dataframe["OriginalImage[Height]"].isna()
        | (metadata_dataframe["OriginalImage[Width]"] <= 0)
        | (metadata_dataframe["OriginalImage[Height]"] <= 0)
    ]

    if len(invalid) > 0:
        raise ValueError(f"Invalid image dimensions for {len(invalid)} metadata rows")


def filter_local_boxes(annotations_dataframe: pd.DataFrame) -> pd.DataFrame:
    has_valid_class = annotations_dataframe["class_name"] != NO_FINDING_LABEL
    has_all_coordinates = annotations_dataframe[COORDINATE_COLUMNS].notna().all(axis=1)
    return annotations_dataframe[has_valid_class & has_all_coordinates].copy()

def filter_boxes_to_configured_classes(
    local_boxes_dataframe: pd.DataFrame,
    class_config: dict,
) -> pd.DataFrame:
    before_count = len(local_boxes_dataframe)

    filtered = local_boxes_dataframe[
        local_boxes_dataframe["class_name"].isin(class_config.keys())
    ].copy()

    removed_count = before_count - len(filtered)
    if removed_count:
        log.info(
            f"Ignored {removed_count} validation boxes from classes not present in class_config_json"
        )

    return filtered

def validate_class_config_covers_observed_classes(
    local_boxes_dataframe: pd.DataFrame,
    class_config: dict,
):
    observed_classes = set(local_boxes_dataframe["class_name"].dropna().unique())
    configured_classes = set(class_config.keys())
    missing_classes = observed_classes - configured_classes

    if missing_classes:
        raise ValueError(
            f"Classes present in validation annotations but missing from class_config_json: "
            f"{sorted(missing_classes)}"
        )


def attach_image_dimensions(
    local_boxes_dataframe: pd.DataFrame,
    metadata_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    image_dimensions = metadata_dataframe.set_index("image_id")[
        ["OriginalImage[Width]", "OriginalImage[Height]"]
    ]

    joined = local_boxes_dataframe.join(image_dimensions, on="image_id")

    missing_dimensions = joined[
        joined["OriginalImage[Width]"].isna()
        | joined["OriginalImage[Height]"].isna()
    ]

    if len(missing_dimensions) > 0:
        missing_image_ids = sorted(missing_dimensions["image_id"].astype(str).unique())
        raise ValueError(
            f"Missing metadata dimensions for {len(missing_image_ids)} images.  "
            f"Examples: {missing_image_ids[:10]}"
        )

    return joined


def clean_and_validate_box_coordinates(local_boxes_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = local_boxes_dataframe.copy()

    for col in COORDINATE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)
    df = df.dropna(
        subset=COORDINATE_COLUMNS + ["OriginalImage[Width]", "OriginalImage[Height]"]
    ).copy()

    dropped_nan = before_drop - len(df)
    if dropped_nan:
        log.warning(f"Dropped {dropped_nan} boxes with non-numeric or missing coordinates")

    df["x_min"] = df["x_min"].clip(lower=0, upper=df["OriginalImage[Width]"])
    df["x_max"] = df["x_max"].clip(lower=0, upper=df["OriginalImage[Width]"])
    df["y_min"] = df["y_min"].clip(lower=0, upper=df["OriginalImage[Height]"])
    df["y_max"] = df["y_max"].clip(lower=0, upper=df["OriginalImage[Height]"])

    valid_geometry = (df["x_max"] > df["x_min"]) & (df["y_max"] > df["y_min"])
    invalid_count = int((~valid_geometry).sum())

    if invalid_count:
        log.warning(f"Dropped {invalid_count} boxes with invalid geometry after clipping")

    return df[valid_geometry].copy()


def xyxy_to_yolo_xywh(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    width = float(np.clip(width, 0.0, 1.0))
    height = float(np.clip(height, 0.0, 1.0))

    return x_center, y_center, width, height


def write_yolo_validation_labels(
    local_boxes_dataframe: pd.DataFrame,
    metadata_dataframe: pd.DataFrame,
    class_config: dict,
    export_yolo_dir: Path,
    all_image_ids: list[str],
):
    export_yolo_dir.mkdir(parents=True, exist_ok=True)

    image_dimensions_by_id = metadata_dataframe.set_index("image_id")[
        ["OriginalImage[Width]", "OriginalImage[Height]"]
    ].to_dict(orient="index")

    boxes_by_image = {
        image_id: image_boxes
        for image_id, image_boxes in local_boxes_dataframe.groupby("image_id")
    }

    number_of_files_written = 0
    number_of_empty_files = 0
    number_of_boxes_written = 0

    for image_id in all_image_ids:
        if image_id not in image_dimensions_by_id:
            raise ValueError(f"No metadata for image_id={image_id}, cannot write YOLO label")

        image_width = image_dimensions_by_id[image_id]["OriginalImage[Width]"]
        image_height = image_dimensions_by_id[image_id]["OriginalImage[Height]"]

        image_boxes = boxes_by_image.get(image_id)
        label_lines = []

        if image_boxes is not None:
            for _, box_row in image_boxes.iterrows():
                class_name = box_row["class_name"]
                class_id = class_config[class_name]["class_id"]

                if class_id < 0:
                    raise ValueError(f"Invalid class_id={class_id} for class {class_name}")

                x_center, y_center, width, height = xyxy_to_yolo_xywh(
                    box_row["x_min"],
                    box_row["y_min"],
                    box_row["x_max"],
                    box_row["y_max"],
                    image_width,
                    image_height,
                )

                if width <= 0.0 or height <= 0.0:
                    continue

                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        output_label_path = export_yolo_dir / f"{image_id}.txt"
        output_label_path.write_text("\n".join(label_lines), encoding="utf-8")

        number_of_files_written += 1
        number_of_boxes_written += len(label_lines)

        if not label_lines:
            number_of_empty_files += 1

    log.info(
        f"YOLO validation labels written: {number_of_files_written} files, "
        f"{number_of_empty_files} empty, {number_of_boxes_written} boxes written into {export_yolo_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Direct VinDr-CXR validation annotation to YOLO label export"
    )
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--class_config_json", required=True, type=Path)
    parser.add_argument("--export_yolo_dir", required=True, type=Path)
    parser.add_argument(
        "--all_images_from",
        choices=["metadata", "annotations"],
        default="metadata",
        help="Use metadata or annotation CSV as the source of all image IDs for empty label creation.",
    )
    parser.add_argument(
        "--allow_class_subset",
        action="store_true",
        help="If set, ignore annotation classes missing from class_config_json instead of raising an error.",
    )

    args = parser.parse_args()

    annotations_dataframe, metadata_dataframe = load_inputs(args.input_csv, args.metadata_csv)
    validate_metadata(metadata_dataframe)

    class_config = load_class_config(args.class_config_json)

    if args.all_images_from == "metadata":
        all_image_ids = sorted(metadata_dataframe["image_id"].astype(str).unique())
    else:
        all_image_ids = sorted(annotations_dataframe["image_id"].astype(str).unique())

    local_boxes_df = filter_local_boxes(annotations_dataframe)

    if args.allow_class_subset:
        local_boxes_df = filter_boxes_to_configured_classes(local_boxes_df, class_config)
    else:
        validate_class_config_covers_observed_classes(local_boxes_df, class_config)

    local_boxes_df = attach_image_dimensions(local_boxes_df, metadata_dataframe)
    local_boxes_df = clean_and_validate_box_coordinates(local_boxes_df)

    log.info(f"Images total:          {len(all_image_ids)}")
    log.info(f"Annotation rows:       {len(annotations_dataframe)}")
    log.info(f"Local box rows:        {len(local_boxes_df)}")
    log.info(f"Images with boxes:     {local_boxes_df['image_id'].nunique()}")
    log.info(f"Images without boxes:  {len(set(all_image_ids) - set(local_boxes_df['image_id'].unique()))}")

    write_yolo_validation_labels(
        local_boxes_dataframe=local_boxes_df,
        metadata_dataframe=metadata_dataframe,
        class_config=class_config,
        export_yolo_dir=args.export_yolo_dir,
        all_image_ids=all_image_ids,
    )


if __name__ == "__main__":
    main()