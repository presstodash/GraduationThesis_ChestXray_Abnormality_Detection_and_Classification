"""
Install:
    pip install fastapi uvicorn ultralytics pillow python-multipart
Run:
    python server.py --yolo_model <path> --classifier_model <path>
http://127.0.0.1:5000/docs
"""

import argparse
import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO, settings
settings.update({"sync": False})

GLOBAL_CLASSIFIER_LABELS = [
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
]

ALL_CLASSIFIER_LABELS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
    "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity",
    "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture",
    "Other lesion", "COPD", "Lung tumor", "Pneumonia", "Tuberculosis",
    "Other diseases",
]

classifier_labels = GLOBAL_CLASSIFIER_LABELS

class ClassificationPrediction(BaseModel):
    class_name: str
    class_id: int
    probability: float

class PredictionBox(BaseModel):
    class_name:   str
    class_id:     int
    confidence:   float
    x_min:        float
    y_min:        float
    x_max:        float
    y_max:        float
    image_width:  int
    image_height: int

class PredictionResponse(BaseModel):
    classifications: list[ClassificationPrediction]
    predictions: list[PredictionBox]
    image_width: int
    image_height: int
    total_detected: int

class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    classifier_loaded: bool

loaded_yolo_model: YOLO | None = None
loaded_classifier_model: nn.Module | None = None
classifier_transform = None
classifier_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

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

def run_classifier(pil_image: Image.Image) -> list[ClassificationPrediction]:
    if loaded_classifier_model is None or classifier_transform is None:
        return []

    image_tensor = classifier_transform(pil_image).unsqueeze(0).to(classifier_device)

    loaded_classifier_model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=classifier_device.type,
            dtype=torch.float16,
            enabled=classifier_device.type == "cuda",
        ):
            logits = loaded_classifier_model(image_tensor)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    results = []
    for class_id, probability in enumerate(probabilities):
        results.append(ClassificationPrediction(
            class_name=classifier_labels[class_id],
            class_id=class_id,
            probability=round(float(probability), 4),
        ))

    results.sort(key=lambda x: x.probability, reverse=True)
    return results

app = FastAPI(
    title="CXR Analyser - Inference API",
    description="YOLO-based chest X-ray abnormality localisation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000", "null"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "yolo_loaded": loaded_yolo_model is not None,
        "classifier_loaded": loaded_classifier_model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    if loaded_yolo_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got: {image.content_type}")

    raw_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {error}")

    original_image_width, original_image_height = pil_image.size

    yolo_results = loaded_yolo_model.predict(pil_image, conf=0.01, verbose=False)

    predictions: list[PredictionBox] = []
    for result in yolo_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_index      = int(box.cls.item())
            confidence_score = float(box.conf.item())
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

            class_name = (
                result.names[class_index]
                if result.names and class_index in result.names
                else f"class_{class_index}"
            )

            predictions.append(PredictionBox(
                class_name=class_name,
                class_id=class_index,
                confidence=round(confidence_score, 4),
                x_min=round(x_min,  2),
                y_min=round(y_min,  2),
                x_max=round(x_max,  2),
                y_max=round(y_max,  2),
                image_width=original_image_width,
                image_height=original_image_height,
            ))

    predictions.sort(key=lambda prediction: prediction.confidence, reverse=True)
    classification_results = run_classifier(pil_image)

    return PredictionResponse(
        classifications=classification_results,
        predictions=predictions,
        image_width=original_image_width,
        image_height=original_image_height,
        total_detected=len(predictions),
    )


def main():
    global loaded_yolo_model, loaded_classifier_model, classifier_transform, classifier_labels

    parser = argparse.ArgumentParser(description="FastAPI YOLO inference server")
    parser.add_argument("--yolo_model", required=True, type=Path)
    parser.add_argument("--classifier_model", required=True, type=Path)
    parser.add_argument(
        "--classifier_architecture",
        default="densenet121",
        choices=["densenet121", "efficientnet_b3", "efficientnet_b4"]
    )
    parser.add_argument("--classifier_image_size", default=224, type=int)
    parser.add_argument("--port",  default=5000, type=int)
    parser.add_argument(
        "--classifier_label_set",
        default="global",
        choices=["global", "all"],
    )
    args = parser.parse_args()

    if not args.yolo_model.exists():
        print(f"ERROR: YOLO weights not found: {args.yolo_model}")
        raise SystemExit(1)

    if not args.classifier_model.exists():
        print(f"ERROR: classifier weights not found: {args.classifier_model}")
        raise SystemExit(1)

    classifier_labels = (
        GLOBAL_CLASSIFIER_LABELS
        if args.classifier_label_set == "global"
        else ALL_CLASSIFIER_LABELS
    )

    print(f"Loading YOLO model from: {args.yolo_model}")
    loaded_yolo_model = YOLO(str(args.yolo_model))

    loaded_classifier_model = build_classifier(
        args.classifier_architecture,
        len(classifier_labels),
    ).to(classifier_device)

    state_dict = torch.load(args.classifier_model, map_location=classifier_device)
    loaded_classifier_model.load_state_dict(state_dict)
    loaded_classifier_model.eval()

    classifier_transform = transforms.Compose([
        transforms.Resize((args.classifier_image_size, args.classifier_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()