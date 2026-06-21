from pathlib import Path
import argparse
import yaml
import json
import numpy as np
import pandas as pd

from ultralytics import YOLO

def load_dataset_yaml(data_yaml_path: Path) -> dict:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required_keys = ["path", "train", "val", "names"]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"data.yaml missing required keys: {missing}")

    return data


def validate_yolo_dataset(data_yaml_path: Path):
    data = load_dataset_yaml(data_yaml_path)

    dataset_root = Path(data["path"])
    train_images_dir = dataset_root / data["train"]
    val_images_dir = dataset_root / data["val"]

    if not train_images_dir.exists():
        raise FileNotFoundError(f"Train images directory not found: {train_images_dir}")
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Val images directory not found: {val_images_dir}")

    names = data["names"]
    if isinstance(names, dict):
        num_classes = len(names)
    elif isinstance(names, list):
        num_classes = len(names)
    else:
        raise ValueError("data.yaml 'names' must be a dict or list")

    for split_name, images_dir in [("train", train_images_dir), ("val", val_images_dir)]:
        labels_dir = Path(str(images_dir).replace("images", "labels"))

        if not labels_dir.exists():
            raise FileNotFoundError(f"{split_name} labels directory not found: {labels_dir}")

        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(images_dir.glob(ext))

        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")

        missing_labels = []
        invalid_labels = []

        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing_labels.append(str(label_path))
                continue

            for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                if len(parts) != 5:
                    invalid_labels.append((str(label_path), line_number, "expected 5 columns"))
                    continue

                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    invalid_labels.append((str(label_path), line_number, "non-numeric value"))
                    continue

                if class_id < 0 or class_id >= num_classes:
                    invalid_labels.append((str(label_path), line_number, f"class_id out of range: {class_id}"))

                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                    invalid_labels.append((str(label_path), line_number, "center outside [0, 1]"))

                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    invalid_labels.append((str(label_path), line_number, "invalid width/height"))

        if missing_labels:
            raise ValueError(
                f"{split_name}: {len(missing_labels)} images are missing label files. "
                f"Examples: {missing_labels[:10]}"
            )

        if invalid_labels:
            raise ValueError(
                f"{split_name}: invalid YOLO labels found. "
                f"Examples: {invalid_labels[:10]}"
            )

        print(f"{split_name}: {len(image_paths)} images validated")

def load_data_yaml(data_yaml_path: Path) -> dict:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "path" not in data or "val" not in data or "names" not in data:
        raise ValueError("data.yaml must contain 'path', 'val', and 'names'")

    return data

def get_class_names(data: dict) -> list[str]:
    names = data["names"]

    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys(), key=lambda x: int(x))]

    if isinstance(names, list):
        return names

    raise ValueError("data.yaml 'names' must be a list or dict")


def get_val_image_and_label_dirs(data_yaml_path: Path) -> tuple[Path, Path, list[str]]:
    data = load_data_yaml(data_yaml_path)
    dataset_root = Path(data["path"])
    val_images_dir = dataset_root / data["val"]

    val_labels_dir = Path(str(val_images_dir).replace("images", "labels"))

    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")

    if not val_labels_dir.exists():
        raise FileNotFoundError(f"Validation labels directory not found: {val_labels_dir}")

    class_names = get_class_names(data)
    return val_images_dir, val_labels_dir, class_names


def list_image_paths(images_dir: Path) -> list[Path]:
    image_paths = []
    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(images_dir.glob(pattern))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(f"No validation images found in {images_dir}")

    return image_paths

def build_image_level_ground_truth(
    image_paths: list[Path],
    labels_dir: Path,
    num_classes: int,
) -> tuple[list[str], np.ndarray]:
    image_ids = []
    y_true = np.zeros((len(image_paths), num_classes), dtype=np.int32)

    for image_index, image_path in enumerate(image_paths):
        image_id = image_path.stem
        image_ids.append(image_id)

        label_path = labels_dir / f"{image_id}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for validation image: {label_path}")

        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            class_id = int(parts[0])

            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"Invalid class_id={class_id} in {label_path}")

            y_true[image_index, class_id] = 1

    return image_ids, y_true


def predict_image_level_scores(
    model: YOLO,
    image_paths: list[Path],
    num_classes: int,
    imgsz: int,
    device,
    conf: float = 0.01,
    iou: float = 0.5,
    max_det: int = 100,
) -> np.ndarray:
    import gc
    import torch

    y_score = np.zeros((len(image_paths), num_classes), dtype=np.float32)

    for image_index, image_path in enumerate(image_paths):
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            max_det=max_det,
            batch=1,
            verbose=False,
            save=False,
            stream=False,
        )

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy().astype(float)

            for class_id, confidence in zip(class_ids, confidences):
                if 0 <= class_id < num_classes:
                    y_score[image_index, class_id] = max(
                        y_score[image_index, class_id],
                        float(confidence),
                    )

        del results
        del result

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    return y_score


def compute_image_level_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn required. Install with pip install scikit-learn"
        ) from exc

    rows = []

    for class_id, class_name in enumerate(class_names):
        positives = int(y_true[:, class_id].sum())
        negatives = int((1 - y_true[:, class_id]).sum())

        row = {
            "class_id": class_id,
            "class_name": class_name,
            "num_positive_images": positives,
            "num_negative_images": negatives,
            "image_level_auroc": np.nan,
            "image_level_average_precision": np.nan,
        }

        if positives > 0 and negatives > 0:
            row["image_level_auroc"] = float(
                roc_auc_score(y_true[:, class_id], y_score[:, class_id])
            )
            row["image_level_average_precision"] = float(
                average_precision_score(y_true[:, class_id], y_score[:, class_id])
            )

        rows.append(row)

    return pd.DataFrame(rows)


def run_post_training_evaluation(
    best_model_path: Path,
    data_yaml_path: Path,
    imgsz: int,
    batch: int,
    device,
    output_dir: Path,
    workers: int,
    image_level_conf: float,
    image_level_iou: float,
    image_level_max_det: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best_model_path))

    detection_metrics = model.val(
        data=str(data_yaml_path),
        imgsz=imgsz,
        batch=batch,
        device=device,
        split="val",
        plots=True,
        verbose=True,
        workers=workers,
    )

    metrics_dict = getattr(detection_metrics, "results_dict", {})

    with open(output_dir / "detection_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    import gc
    import torch

    del detection_metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_images_dir, val_labels_dir, class_names = get_val_image_and_label_dirs(data_yaml_path)
    image_paths = list_image_paths(val_images_dir)

    image_ids, y_true = build_image_level_ground_truth(
        image_paths=image_paths,
        labels_dir=val_labels_dir,
        num_classes=len(class_names),
    )

    y_score = predict_image_level_scores(
        model=model,
        image_paths=image_paths,
        num_classes=len(class_names),
        imgsz=imgsz,
        device=device,
        conf=image_level_conf,
        iou=image_level_iou,
        max_det=image_level_max_det,
    )

    auroc_df = compute_image_level_auroc(
        y_true=y_true,
        y_score=y_score,
        class_names=class_names,
    )

    auroc_df.to_csv(output_dir / "image_level_auroc_by_class.csv", index=False)

    image_scores_df = pd.DataFrame({"image_id": image_ids})
    for class_id, class_name in enumerate(class_names):
        safe_name = class_name.replace("/", "_").replace(" ", "_")
        image_scores_df[f"true_{safe_name}"] = y_true[:, class_id]
        image_scores_df[f"score_{safe_name}"] = y_score[:, class_id]

    image_scores_df.to_csv(output_dir / "image_level_scores.csv", index=False)

    print("\n=== Detection metrics ===")
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    print("\n=== Image-level AUROC by class ===")
    print(
        auroc_df.sort_values("image_level_auroc", ascending=False)
        .to_string(index=False)
    )

def save_run_config(args, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def clear_yolo_label_cache(data_yaml_path: Path):
    data = load_dataset_yaml(data_yaml_path)
    dataset_root = Path(data["path"])

    for split_key in ["train", "val"]:
        images_dir = dataset_root / data[split_key]
        labels_dir = Path(str(images_dir).replace("images", "labels"))

        cache_candidates = [
            labels_dir.with_suffix(".cache"),
            labels_dir / "labels.cache",
        ]

        for cache_path in cache_candidates:
            if cache_path.exists():
                cache_path.unlink()
                print(f"Deleted stale YOLO cache: {cache_path}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLO localizer on VinDr-CXR derived labels")
    parser.add_argument("--data_yaml", required=True, type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=1024, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/vindr_localizer")
    parser.add_argument("--name", default="yolo_localizer")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--image_level_conf", default=0.01, type=float)
    parser.add_argument("--image_level_iou", default=0.5, type=float)
    parser.add_argument("--mosaic", default=0.0, type=float)
    parser.add_argument("--mixup", default=0.0, type=float)
    parser.add_argument("--copy_paste", default=0.0, type=float)
    parser.add_argument("--degrees", default=0.0, type=float)
    parser.add_argument("--translate", default=0.05, type=float)
    parser.add_argument("--scale", default=0.2, type=float)
    parser.add_argument("--fliplr", default=0.0, type=float)
    parser.add_argument("--clear_label_cache", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--weights", default=None, type=Path)
    parser.add_argument("--image_level_max_det", default=100, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr0", default=None, type=float)
    parser.add_argument("--lrf", default=None, type=float)
    parser.add_argument("--optimizer", default="auto")
    args = parser.parse_args()

    if args.clear_label_cache:
        clear_yolo_label_cache(args.data_yaml)

    if args.eval_only:
        if args.weights is None:
            raise ValueError("--weights is required when --eval_only is used")

        run_post_training_evaluation(
            best_model_path=args.weights,
            data_yaml_path=args.data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_dir=Path(args.project) / args.name / "post_train_eval",
            workers=args.workers,
            image_level_conf=args.image_level_conf,
            image_level_iou=args.image_level_iou,
            image_level_max_det=args.image_level_max_det,
        )
        return

    validate_yolo_dataset(args.data_yaml)

    if args.validate_only:
        print("Dataset validation passed.")
        return

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        task="detect",
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    if args.resume:
        train_kwargs["resume"] = True
    train_kwargs["optimizer"] = args.optimizer
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf


    train_results = model.train(**train_kwargs)

    run_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
    best_model_path = run_dir / "weights" / "best.pt"

    save_run_config(args, run_dir / "training_script_args.json")

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    run_post_training_evaluation(
        best_model_path=best_model_path,
        data_yaml_path=args.data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_dir=run_dir / "post_train_eval",
        workers=args.workers,
        image_level_conf=args.image_level_conf,
        image_level_iou=args.image_level_iou,
        image_level_max_det=args.image_level_max_det,
    )

if __name__ == "__main__":
    main()