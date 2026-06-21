import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


GLOBAL_LABEL_COLUMNS = [
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
]


class ChestXrayClassificationDataset(Dataset):
    def __init__(self, image_dir: Path, labels_dataframe: pd.DataFrame, image_transform):
        self.image_dir = image_dir
        self.labels_dataframe = labels_dataframe.reset_index(drop=True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.labels_dataframe)

    def __getitem__(self, index):
        row = self.labels_dataframe.iloc[index]
        image_path = self.image_dir / f"{row['image_id']}.png"

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        label_vector = torch.tensor(
            row[GLOBAL_LABEL_COLUMNS].values.astype(np.float32),
            dtype=torch.float32,
        )

        return image_tensor, label_vector, row["image_id"]


def build_classifier(architecture: str, number_of_output_classes: int) -> nn.Module:
    if architecture == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, number_of_output_classes)
        return model

    if architecture == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, number_of_output_classes)
        return model

    if architecture == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, number_of_output_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")


def evaluate_model(model, dataloader, device):
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    all_image_ids = []

    with torch.no_grad():
        for image_batch, label_batch, image_ids in tqdm(dataloader, desc="Evaluating"):
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(image_batch)
                loss = loss_function(logits, label_batch)

            probabilities = torch.sigmoid(logits)

            total_loss += loss.item()
            all_labels.append(label_batch.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_image_ids.extend(image_ids)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probabilities, axis=0)
    val_loss = total_loss / len(dataloader)

    return val_loss, y_true, y_prob, all_image_ids


def compute_metrics(y_true, y_prob, threshold=0.5):
    rows = []

    for class_index, class_name in enumerate(GLOBAL_LABEL_COLUMNS):
        true_class = y_true[:, class_index]
        prob_class = y_prob[:, class_index]
        pred_class = (prob_class >= threshold).astype(int)

        positives = int(true_class.sum())
        negatives = int(len(true_class) - positives)

        if len(np.unique(true_class)) >= 2:
            auroc = roc_auc_score(true_class, prob_class)
            ap = average_precision_score(true_class, prob_class)
        else:
            auroc = np.nan
            ap = np.nan

        rows.append({
            "class_name": class_name,
            "positives": positives,
            "negatives": negatives,
            "auroc": auroc,
            "average_precision": ap,
            "precision_at_0_5": precision_score(true_class, pred_class, zero_division=0),
            "recall_at_0_5": recall_score(true_class, pred_class, zero_division=0),
            "f1_at_0_5": f1_score(true_class, pred_class, zero_division=0),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", required=True, type=Path)
    parser.add_argument("--test_labels_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument(
        "--model",
        action="append",
        nargs=4,
        metavar=("NAME", "ARCHITECTURE", "IMAGE_SIZE", "CHECKPOINT"),
        required=True,
        help="Example: --model densenet121 densenet121 224 path/to/model.pth",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataframe = pd.read_csv(args.test_labels_csv)

    missing_columns = [c for c in GLOBAL_LABEL_COLUMNS if c not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing global label columns in CSV: {missing_columns}")

    summary_rows = []

    for model_name, architecture, image_size_string, checkpoint_path_string in args.model:
        image_size = int(image_size_string)
        checkpoint_path = Path(checkpoint_path_string)

        print(f"\nEvaluating model: {model_name}")
        print(f"Architecture: {architecture}")
        print(f"Image size: {image_size}")
        print(f"Checkpoint: {checkpoint_path}")

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        dataset = ChestXrayClassificationDataset(
            args.test_image_dir,
            dataframe,
            transform,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = build_classifier(architecture, len(GLOBAL_LABEL_COLUMNS)).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        val_loss, y_true, y_prob, image_ids = evaluate_model(model, dataloader, device)

        per_class_metrics = compute_metrics(y_true, y_prob)
        mean_auroc = per_class_metrics["auroc"].mean()
        mean_ap = per_class_metrics["average_precision"].mean()
        mean_f1 = per_class_metrics["f1_at_0_5"].mean()

        per_class_path = args.output_dir / f"{model_name}_per_class_metrics.csv"
        per_class_metrics.to_csv(per_class_path, index=False)

        predictions_dataframe = pd.DataFrame({"image_id": image_ids})
        for class_index, class_name in enumerate(GLOBAL_LABEL_COLUMNS):
            predictions_dataframe[f"true_{class_name}"] = y_true[:, class_index]
            predictions_dataframe[f"prob_{class_name}"] = y_prob[:, class_index]

        predictions_path = args.output_dir / f"{model_name}_predictions.csv"
        predictions_dataframe.to_csv(predictions_path, index=False)

        summary_rows.append({
            "model": model_name,
            "architecture": architecture,
            "image_size": image_size,
            "val_loss": val_loss,
            "mean_auc_roc": mean_auroc,
            "mean_average_precision": mean_ap,
            "mean_f1_at_0_5": mean_f1,
            "checkpoint": str(checkpoint_path),
        })

        print(f"Val loss: {val_loss:.4f}")
        print(f"Mean AUROC: {mean_auroc:.4f}")
        print(f"Mean AP: {mean_ap:.4f}")
        print(f"Mean F1@0.5: {mean_f1:.4f}")

    summary_dataframe = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "summary_metrics.csv"
    summary_dataframe.to_csv(summary_path, index=False)

    print(f"\nSaved summary into: {summary_path}")


if __name__ == "__main__":
    main()