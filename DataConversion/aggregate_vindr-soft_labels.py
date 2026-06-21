import argparse
import sys
from pathlib import Path
 
import pandas as pd
 
 
DISEASE_LABEL_COLUMNS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
    "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity",
    "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture",
    "Other lesion", "COPD", "Lung tumor", "Pneumonia", "Tuberculosis",
    "Other diseases",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  "-i", required=True, type=Path)
    parser.add_argument("--output_csv", "-o", required=True, type=Path)
    args = parser.parse_args()
 
    if not args.input_csv.exists():
        print(f"ERROR: input CSV not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
 
    raw_dataframe = pd.read_csv(args.input_csv)
 
    print(f"Loaded {len(raw_dataframe)} rows from {args.input_csv}")

    soft_labels_dataframe = (
        raw_dataframe
        .groupby("image_id")[DISEASE_LABEL_COLUMNS]
        .mean()
        .reset_index()
    )
 
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    soft_labels_dataframe.to_csv(args.output_csv, index=False)
 
    print(f"Saved {len(soft_labels_dataframe)} rows into {args.output_csv}")

    full_agreement = (
        soft_labels_dataframe[DISEASE_LABEL_COLUMNS]
        .apply(lambda column: (column == 1.0).mean() + (column == 0.0).mean())
        .mean()
    )
    print(f"Average full agreement rate across classes equals {full_agreement:.1%}")
 
 
if __name__ == "__main__":
    main()
