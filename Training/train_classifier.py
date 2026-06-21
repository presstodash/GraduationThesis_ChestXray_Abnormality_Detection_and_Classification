import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ALL_LABEL_COLUMNS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
    "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity",
    "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture",
    "Other lesion", "COPD", "Lung tumor", "Pneumonia", "Tuberculosis",
    "Other diseases",
]

GLOBAL_LABEL_COLUMNS = [
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
]

class ChestXrayClassificationDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        labels_dataframe: pd.DataFrame,
        image_transform,
        label_columns: list[str],
    ):
        self.image_dir = image_dir
        self.labels_dataframe = labels_dataframe.reset_index(drop=True)
        self.image_transform = image_transform
        self.label_columns = label_columns

    def __len__(self):
        return len(self.labels_dataframe)

    def __getitem__(self, index):
        row = self.labels_dataframe.iloc[index]

        image_path = self.image_dir / f"{row['image_id']}.png"
        
        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(pil_image)

        label_vector = torch.tensor(
            row[self.label_columns].values.astype(np.float32),
            dtype=torch.float32,
        )

        return image_tensor, label_vector


def build_classifier(architecture: str, number_of_output_classes: int) -> nn.Module:
    if architecture == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, number_of_output_classes)
        return model

    if architecture == "efficientnet_b3":
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, number_of_output_classes)
        return model

    if architecture == "efficientnet_b4":
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, number_of_output_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")

def get_default_image_size(architecture: str) -> int:
    if architecture == "densenet121":
        return 224
    if architecture == "efficientnet_b3":
        return 300
    if architecture == "efficientnet_b4":
        return 380
    raise ValueError(f"Unsupported architecture: {architecture}")

def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for image_batch, label_batch in tqdm(dataloader, desc="  Train", leave=False):
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda"
        ):
            logit_predictions = model(image_batch)
            batch_loss = loss_function(logit_predictions, label_batch)

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += batch_loss.item()

    return total_loss / len(dataloader)

def run_validation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    number_of_classes: int,
) -> tuple[float, float]:
    """Returns (validation_loss, mean_auc_roc)."""
    model.eval()
    total_loss = 0.0
    all_true_labels = []
    all_predicted_probabilities = []

    with torch.no_grad():
        for image_batch, label_batch in tqdm(dataloader, desc="  Val  ", leave=False):
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):
                logit_predictions = model(image_batch)
                batch_loss = loss_function(logit_predictions, label_batch)

            total_loss += batch_loss.item()
            all_true_labels.append(label_batch.cpu().numpy())
            all_predicted_probabilities.append(
                torch.sigmoid(logit_predictions).cpu().numpy()
            )

    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_predicted_probabilities = np.concatenate(all_predicted_probabilities, axis=0)

    
    per_class_auc_scores = []
    for class_index in range(number_of_classes):
        unique_labels_in_class = np.unique(all_true_labels[:, class_index])
        if len(unique_labels_in_class) < 2:
            continue
        class_auc = roc_auc_score(
            all_true_labels[:, class_index],
            all_predicted_probabilities[:, class_index],
        )
        per_class_auc_scores.append(class_auc)

    mean_auc_roc = float(np.mean(per_class_auc_scores)) if per_class_auc_scores else 0.0
    validation_loss = total_loss / len(dataloader)

    return validation_loss, mean_auc_roc

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN classifier for multilabel chest X-ray classification."
    )
    parser.add_argument("--train_image_dir",  required=True,  type=Path)
    parser.add_argument("--train_labels_csv", required=True,  type=Path)
    parser.add_argument("--test_image_dir",   required=True,  type=Path)
    parser.add_argument("--test_labels_csv",  required=True,  type=Path)
    parser.add_argument("--output_dir",    "-o", required=True,  type=Path)
    parser.add_argument("--epochs",              default=50,     type=int)
    parser.add_argument("--batch_size",          default=16,     type=int)
    parser.add_argument("--lr",                  default=1e-4,   type=float,
                        help="Learning rate (default: 1e-4).")
    parser.add_argument("--num_workers",         default=4,      type=int,
                        help="DataLoader worker processes (default: 4).")
    parser.add_argument("--seed",                default=42,     type=int)
    parser.add_argument(
        "--architecture",
        default="densenet121",
        choices=["densenet121", "efficientnet_b3", "efficientnet_b4"],
        help="Classification architecture to train."
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help="Input image size. If not set, uses architecture default."
    )
    parser.add_argument(
        "--label_set",
        default="all",
        choices=["all", "global"],
        help="Which label set to train on: all labels or only global disease labels."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_dataframe   = pd.read_csv(args.train_labels_csv)
    validation_dataframe = pd.read_csv(args.test_labels_csv)

    if args.label_set == "all":
        label_columns = ALL_LABEL_COLUMNS
    elif args.label_set == "global":
        label_columns = GLOBAL_LABEL_COLUMNS
    else:
        raise ValueError(f"Unsupported label_set: {args.label_set}")

    missing_train = [c for c in label_columns if c not in training_dataframe.columns]
    missing_val = [c for c in label_columns if c not in validation_dataframe.columns]

    if missing_train:
        raise ValueError(f"Missing columns in train CSV: {missing_train}")
    if missing_val:
        raise ValueError(f"Missing columns in test CSV: {missing_val}")

    number_of_classes = len(label_columns)

    print(f"Label set: {args.label_set}")
    print(f"Classes  : {label_columns}")

 
    print(f"Training samples  : {len(training_dataframe)}")
    print(f"Validation samples: {len(validation_dataframe)}")    
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    image_size = args.image_size
    if image_size is None:
        image_size = get_default_image_size(args.architecture)

    print(f"Architecture: {args.architecture}")
    print(f"Image size  : {image_size}x{image_size}")

    training_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    
    training_dataset = ChestXrayClassificationDataset(
        args.train_image_dir, training_dataframe, training_transform, label_columns
    )
    validation_dataset = ChestXrayClassificationDataset(
        args.test_image_dir, validation_dataframe, validation_transform, label_columns
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    model = build_classifier(args.architecture, number_of_classes).to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    gradient_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    min_auc_delta = 0.001
    best_mean_auc = 0.0
    best_val_loss = float("inf")
    best_model_checkpoint_path = args.output_dir / f"best_model_{args.architecture}_{args.label_set}_{image_size}.pth"
    training_log_rows = []

    for epoch_number in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch_number}/{args.epochs}")

        training_loss = run_training_epoch(
            model, training_dataloader, loss_function,
            optimizer, gradient_scaler, device
        )
        validation_loss, mean_auc_roc = run_validation_epoch(
            model, validation_dataloader, loss_function, device, number_of_classes
        )

        lr_scheduler.step(mean_auc_roc)

        print(f"  Train loss: {training_loss:.4f}")
        print(f"  Val loss  : {validation_loss:.4f}  |  Mean AUC-ROC: {mean_auc_roc:.4f}")

        is_better_auc = mean_auc_roc > best_mean_auc + min_auc_delta
        is_same_auc_better_loss = (
            abs(mean_auc_roc - best_mean_auc) <= min_auc_delta
            and validation_loss < best_val_loss
        )

        if is_better_auc or is_same_auc_better_loss:
            best_mean_auc = mean_auc_roc
            best_val_loss = validation_loss
            torch.save(model.state_dict(), best_model_checkpoint_path)
            print(
                f"[NOTIFICATION] New best model saved "
                f"(AUC-ROC: {best_mean_auc:.4f}, Val loss: {best_val_loss:.4f})"
            )

        training_log_rows.append({
            "epoch":          epoch_number,
            "train_loss":     round(training_loss,  4),
            "val_loss":       round(validation_loss, 4),
            "mean_auc_roc":   round(mean_auc_roc,   4),
            "architecture": args.architecture,
            "image_size": image_size,
            "label_set": args.label_set,
            "num_classes": number_of_classes,
            "best_val_loss_so_far": round(best_val_loss, 4),
        })

    training_log_path = args.output_dir / "training_log.csv"
    pd.DataFrame(training_log_rows).to_csv(training_log_path, index=False)
    print(f"\nTraining complete. Best AUC-ROC: {best_mean_auc:.4f}")
    print(f"Best model saved into {best_model_checkpoint_path}.")
    print(f"Training log saved into {training_log_path}.")

if __name__ == "__main__":
    main()