import argparse
import sys
from pathlib import Path

import pandas as pd

def get_sorted_class_names(train_csv_path: Path, test_csv_path: Path) -> list[str]:
    train_class_names = pd.read_csv(train_csv_path)["class_name"].unique().tolist()
    test_class_names  = pd.read_csv(test_csv_path)["class_name"].unique().tolist()
    all_class_names   = sorted(set(train_class_names + test_class_names))
    return all_class_names

def convert_csv_to_yolo_files(
    annotations_dataframe: pd.DataFrame,
    image_dir: Path,
    output_labels_dir: Path,
    class_name_to_index: dict[str, int],
    image_width: int,
    image_height: int,
):
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_id, annotations in annotations_dataframe.groupby("image_id"):
        path = image_dir / f"{image_id}.png"
        if not path.exists():
            print(f"WARNING: image not found: {path}", file=sys.stderr)
            continue

        label_file_path = output_labels_dir / f"{image_id}.txt"
        label_lines = []

        for _, row in annotations.iterrows():
            class_index = class_name_to_index.get(row["class_name"])
            if class_index is None:
                continue

            x_center = ((row["x_min"] + row["x_max"]) / 2) / image_width
            y_center = ((row["y_min"] + row["y_max"]) / 2) / image_height
            width_new    = (row["x_max"] - row["x_min"]) / image_width
            height_new   = (row["y_max"] - row["y_min"]) / image_height

            label_lines.append(
                f"{class_index} {x_center:.6f} {y_center:.6f} "
                f"{width_new:.6f} {height_new:.6f}"
            )

        label_file_path.write_text("\n".join(label_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",        required=True,  type=Path)
    parser.add_argument("--test_csv",         required=True,  type=Path)
    parser.add_argument("--train_image_dir",  required=True,  type=Path)
    parser.add_argument("--test_image_dir",   required=True,  type=Path)
    parser.add_argument("--output_dir",       required=True,  type=Path)
    parser.add_argument("--image_width",      default=1024,   type=int)
    parser.add_argument("--image_height",     default=1024,   type=int)
    args = parser.parse_args()

    for required_path in [args.train_csv, args.test_csv, args.train_image_dir, args.test_image_dir]:
        if not required_path.exists():
            print(f"ERROR: path not found: {required_path}", file=sys.stderr)
            sys.exit(1)

    all_class_names    = get_sorted_class_names(args.train_csv, args.test_csv)
    class_name_to_index = {class_name: index for index, class_name in enumerate(all_class_names)}

    train_annotations_dataframe = pd.read_csv(args.train_csv)
    test_annotations_dataframe  = pd.read_csv(args.test_csv)

    train_labels_output_dir = args.output_dir / "labels" / "train"
    test_labels_output_dir  = args.output_dir / "labels" / "test"

    convert_csv_to_yolo_files(
        train_annotations_dataframe,
        args.train_image_dir,
        train_labels_output_dir,
        class_name_to_index,
        args.image_width,
        args.image_height,
    )
    print(f"Train labels saved -> {train_labels_output_dir}")

    convert_csv_to_yolo_files(
        test_annotations_dataframe,
        args.test_image_dir,
        test_labels_output_dir,
        class_name_to_index,
        args.image_width,
        args.image_height,
    )
    print(f"Test labels saved  -> {test_labels_output_dir}")

    dataset_yaml_content = f"""path: {args.output_dir.resolve()}
train: images/train
val:   images/test

nc: {len(all_class_names)}
names: {all_class_names}
"""
    yaml_output_path = args.output_dir / "dataset.yaml"
    yaml_output_path.write_text(dataset_yaml_content)
    print(f"dataset.yaml saved -> {yaml_output_path}")

    classes_txt_content = "\n".join(all_class_names)
    classes_txt_path = args.output_dir / "classes.txt"
    classes_txt_path.write_text(classes_txt_content)
    print(f"classes.txt saved  -> {classes_txt_path}")

if __name__ == "__main__":
    main()