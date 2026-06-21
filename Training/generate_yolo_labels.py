import argparse
import json
import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NO_FINDING_LABEL               = "No finding"
REQUIRED_ANNOTATION_COLUMNS    = ["image_id", "rad_id", "class_name", "x_min", "y_min", "x_max", "y_max"]
REQUIRED_METADATA_COLUMNS      = ["image_id", "OriginalImage[Width]", "OriginalImage[Height]"]
COORDINATE_COLUMNS             = ["x_min", "y_min", "x_max", "y_max"]

DEFAULT_IOU_THRESHOLD          = 0.35
DEFAULT_CENTER_DISTANCE_FACTOR = 1.5
INTRA_READER_DEDUP_IOU         = 0.85

INITIAL_SENSITIVITY            = 0.75
INITIAL_SPECIFICITY            = 0.90
INITIAL_LOCALIZATION_WEIGHT    = 1.0
EM_SMOOTHING_EPS               = 1e-3
SENSITIVITY_MIN                = 0.05
SENSITIVITY_MAX                = 0.99
SPECIFICITY_MIN                = 0.05
SPECIFICITY_MAX                = 0.99

AGGREGATION_METHOD_LABEL       = "candidate_level_em_bdc_inspired"

SOFT_CONSENSUS_COLUMNS = [
    "image_id",
    "class_name",
    "class_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "posterior_prob",
    "raw_support",
    "num_readers",
    "supporting_rad_ids",
    "target_weight",
    "aggregation_method",
]

READER_STATS_COLUMNS = [
    "rad_id",
    "class_name",
    "sensitivity",
    "candidate_specificity",
    "num_supported_candidates",
    "num_seen_candidates",
    "is_reliable_estimate",
    "parameter_source",
]

@dataclass
class CandidateLesion:
    image_id:             str
    class_name:           str
    class_id:             int
    boxes:                list
    supporting_rad_ids:   list
    num_readers:          int
    posterior_prob:       float
    aggregated_box_xyxy:  tuple = field(default=(0.0, 0.0, 0.0, 0.0))

def validate_columns(dataframe: pd.DataFrame, required_columns: list[str], source_name: str):
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{source_name} is missing columns: {missing_columns}")


def load_class_config(class_config_json_path: Path) -> dict:
    if not class_config_json_path.exists():
        raise FileNotFoundError(f"class_config_json not found: {class_config_json_path}")

    with open(class_config_json_path) as json_file:
        raw_config = json.load(json_file)

    class_config = raw_config.get("classes")
    if not isinstance(class_config, dict) or not class_config:
        raise ValueError("class_config_json must contain a non-empty 'classes' object")

    for class_name, cfg in class_config.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Configuration for class '{class_name}' must be an object")
        if "class_id" not in cfg:
            raise ValueError(f"Missing class_id for class '{class_name}' in class_config_json")

    return class_config

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
            f"Ignored {removed_count} local boxes from classes not included in class_config_json"
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
            f"Classes are there in annotations but missing from class_config_json: {sorted(missing_classes)}"
        )


def load_inputs(
    input_csv_path: Path,
    metadata_csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations_dataframe = pd.read_csv(input_csv_path)
    metadata_dataframe    = pd.read_csv(metadata_csv_path)

    validate_columns(annotations_dataframe, REQUIRED_ANNOTATION_COLUMNS, "input_csv")
    validate_columns(metadata_dataframe,    REQUIRED_METADATA_COLUMNS,    "metadata_csv")

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

def build_readers_by_image(annotations_dataframe: pd.DataFrame) -> dict[str, list]:
    return (
        annotations_dataframe
        .groupby("image_id")["rad_id"]
        .apply(lambda radiologist_ids: sorted(set(radiologist_ids), key=str))
        .to_dict()
    )


def filter_local_boxes(annotations_dataframe: pd.DataFrame) -> pd.DataFrame:
    has_valid_class = annotations_dataframe["class_name"] != NO_FINDING_LABEL
    has_all_coordinates = annotations_dataframe[COORDINATE_COLUMNS].notna().all(axis=1)
    return annotations_dataframe[has_valid_class & has_all_coordinates].copy()


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
            f"Missing metadata dimensions for {len(missing_image_ids)} images."
            f"Examples: {missing_image_ids[:10]}"
        )

    return joined


def clean_and_validate_box_coordinates(local_boxes_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = local_boxes_dataframe.copy()

    for col in COORDINATE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=COORDINATE_COLUMNS + ["OriginalImage[Width]", "OriginalImage[Height]"]).copy()
    dropped_nan = before_drop - len(df)
    if dropped_nan:
        log.warning(f"Dropped {dropped_nan} local boxes with non-numeric or missing coordinates")

    df["x_min"] = df["x_min"].clip(lower=0, upper=df["OriginalImage[Width]"])
    df["x_max"] = df["x_max"].clip(lower=0, upper=df["OriginalImage[Width]"])
    df["y_min"] = df["y_min"].clip(lower=0, upper=df["OriginalImage[Height]"])
    df["y_max"] = df["y_max"].clip(lower=0, upper=df["OriginalImage[Height]"])

    valid_geometry = (df["x_max"] > df["x_min"]) & (df["y_max"] > df["y_min"])
    invalid_count = int((~valid_geometry).sum())
    if invalid_count:
        log.warning(f"Dropped {invalid_count} local boxes with invalid geometry after clipping")

    df = df[valid_geometry].copy()
    return df


def normalize_xyxy_boxes(local_boxes_dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe_with_normalized = local_boxes_dataframe.copy()
    dataframe_with_normalized["x_min_norm"] = (
        dataframe_with_normalized["x_min"] / dataframe_with_normalized["OriginalImage[Width]"]
    )
    dataframe_with_normalized["y_min_norm"] = (
        dataframe_with_normalized["y_min"] / dataframe_with_normalized["OriginalImage[Height]"]
    )
    dataframe_with_normalized["x_max_norm"] = (
        dataframe_with_normalized["x_max"] / dataframe_with_normalized["OriginalImage[Width]"]
    )
    dataframe_with_normalized["y_max_norm"] = (
        dataframe_with_normalized["y_max"] / dataframe_with_normalized["OriginalImage[Height]"]
    )
    return dataframe_with_normalized

def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    intersection_x_min = max(box_a[0], box_b[0])
    intersection_y_min = max(box_a[1], box_b[1])
    intersection_x_max = min(box_a[2], box_b[2])
    intersection_y_max = min(box_a[3], box_b[3])

    intersection_width  = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    intersection_area   = intersection_width * intersection_height

    area_of_box_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_of_box_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area    = area_of_box_a + area_of_box_b - intersection_area

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area

def compute_center_distance(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    center_a_x = (box_a[0] + box_a[2]) / 2.0
    center_a_y = (box_a[1] + box_a[3]) / 2.0
    center_b_x = (box_b[0] + box_b[2]) / 2.0
    center_b_y = (box_b[1] + box_b[3]) / 2.0
    return math.sqrt((center_a_x - center_b_x) ** 2 + (center_a_y - center_b_y) ** 2)


def boxes_match(
    box_a: tuple,
    box_b: tuple,
    iou_threshold: float,
    center_distance_factor: float,
    use_center_distance: bool,
) -> bool:
    iou = compute_iou(box_a, box_b)
    if iou >= iou_threshold:
        return True

    if not use_center_distance:
        return False

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    if area_a <= 0.0 or area_b <= 0.0:
        return False

    mean_box_area = (area_a + area_b) / 2.0
    distance_limit = center_distance_factor * math.sqrt(mean_box_area)
    center_distance = compute_center_distance(box_a, box_b)

    return center_distance <= distance_limit


def get_normalized_box_tuple(box_record: dict) -> tuple[float, float, float, float]:
    return (
        float(box_record["x_min_norm"]),
        float(box_record["y_min_norm"]),
        float(box_record["x_max_norm"]),
        float(box_record["y_max_norm"]),
    )

def deduplicate_within_reader(
    reader_boxes: list[dict],
) -> list[dict]:
    if len(reader_boxes) <= 1:
        return reader_boxes

    merged_flags = [False] * len(reader_boxes)
    deduplicated_boxes = []

    for index_a, box_a_record in enumerate(reader_boxes):
        if merged_flags[index_a]:
            continue

        box_a_coords = get_normalized_box_tuple(box_a_record)
        boxes_to_merge = [box_a_record]

        for index_b in range(index_a + 1, len(reader_boxes)):
            if merged_flags[index_b]:
                continue
            box_b_record = reader_boxes[index_b]
            box_b_coords = get_normalized_box_tuple(box_b_record)
            if compute_iou(box_a_coords, box_b_coords) > INTRA_READER_DEDUP_IOU:
                boxes_to_merge.append(box_b_record)
                merged_flags[index_b] = True

        if len(boxes_to_merge) == 1:
            deduplicated_boxes.append(box_a_record)
        else:
            merged_record = box_a_record.copy()
            merged_record["x_min_norm"] = float(np.mean([b["x_min_norm"] for b in boxes_to_merge]))
            merged_record["y_min_norm"] = float(np.mean([b["y_min_norm"] for b in boxes_to_merge]))
            merged_record["x_max_norm"] = float(np.mean([b["x_max_norm"] for b in boxes_to_merge]))
            merged_record["y_max_norm"] = float(np.mean([b["y_max_norm"] for b in boxes_to_merge]))
            merged_record["x_min"]      = float(np.mean([b["x_min"] for b in boxes_to_merge]))
            merged_record["y_min"]      = float(np.mean([b["y_min"] for b in boxes_to_merge]))
            merged_record["x_max"]      = float(np.mean([b["x_max"] for b in boxes_to_merge]))
            merged_record["y_max"]      = float(np.mean([b["y_max"] for b in boxes_to_merge]))
            deduplicated_boxes.append(merged_record)

    return deduplicated_boxes



def build_candidate_clusters(
    image_class_boxes: list[dict],
    iou_threshold: float,
    center_distance_factor: float,
    use_center_distance: bool,
) -> list[list[dict]]:
    if not image_class_boxes:
        return []

    unused_indices = set(range(len(image_class_boxes)))
    clusters = []

    while unused_indices:
        seed_index = min(unused_indices)
        seed_box = image_class_boxes[seed_index]
        seed_coords = get_normalized_box_tuple(seed_box)

        cluster_indices = [seed_index]
        cluster_reader_ids = {seed_box["rad_id"]}
        unused_indices.remove(seed_index)

        candidate_matches = []

        for other_index in list(unused_indices):
            other_box = image_class_boxes[other_index]

            if other_box["rad_id"] in cluster_reader_ids:
                continue

            other_coords = get_normalized_box_tuple(other_box)

            if boxes_match(
                seed_coords,
                other_coords,
                iou_threshold,
                center_distance_factor,
                use_center_distance,
            ):
                match_iou = compute_iou(seed_coords, other_coords)
                match_distance = compute_center_distance(seed_coords, other_coords)
                candidate_matches.append((other_index, match_iou, match_distance))

        candidate_matches.sort(key=lambda item: (-item[1], item[2]))

        for other_index, _, _ in candidate_matches:
            if other_index not in unused_indices:
                continue

            other_box = image_class_boxes[other_index]
            if other_box["rad_id"] in cluster_reader_ids:
                continue

            other_coords = get_normalized_box_tuple(other_box)
            matches_existing_cluster = False

            for cluster_index in cluster_indices:
                cluster_coords = get_normalized_box_tuple(image_class_boxes[cluster_index])
                if boxes_match(
                    cluster_coords,
                    other_coords,
                    iou_threshold,
                    center_distance_factor,
                    use_center_distance,
                ):
                    matches_existing_cluster = True
                    break

            if not matches_existing_cluster:
                continue

            cluster_indices.append(other_index)
            cluster_reader_ids.add(other_box["rad_id"])
            unused_indices.remove(other_index)

        clusters.append([image_class_boxes[index] for index in cluster_indices])

    return clusters

def initialize_reader_parameters(
    all_reader_ids: list[str],
    all_class_names: list[str],
) -> tuple[dict, dict, dict]:
    sensitivity_by_reader_and_class          = {}
    specificity_by_reader_and_class          = {}
    localization_weight_by_reader_and_class  = {}

    for reader_id in all_reader_ids:
        for class_name in all_class_names:
            key = (reader_id, class_name)
            sensitivity_by_reader_and_class[key]         = INITIAL_SENSITIVITY
            specificity_by_reader_and_class[key]         = INITIAL_SPECIFICITY
            localization_weight_by_reader_and_class[key] = INITIAL_LOCALIZATION_WEIGHT

    return (
        sensitivity_by_reader_and_class,
        specificity_by_reader_and_class,
        localization_weight_by_reader_and_class,
    )


def sigmoid_from_logit(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def run_e_step(
    candidate: CandidateLesion,
    readers_by_image: dict[str, list],
    sensitivity_by_reader_and_class: dict,
    specificity_by_reader_and_class: dict,
    prior_probability: float = 0.5,
) -> float:
    all_readers_for_image  = readers_by_image.get(candidate.image_id, [])
    supporting_reader_set  = set(candidate.supporting_rad_ids)

    log_probability_lesion_exists      = math.log(prior_probability + 1e-12)
    log_probability_lesion_not_exists  = math.log(1.0 - prior_probability + 1e-12)

    for reader_id in all_readers_for_image:
        key           = (reader_id, candidate.class_name)
        sensitivity   = sensitivity_by_reader_and_class.get(key, INITIAL_SENSITIVITY)
        specificity   = specificity_by_reader_and_class.get(key, INITIAL_SPECIFICITY)

        if reader_id in supporting_reader_set:
            log_probability_lesion_exists     += math.log(sensitivity + 1e-12)
            log_probability_lesion_not_exists += math.log(1.0 - specificity + 1e-12)
        else:
            log_probability_lesion_exists     += math.log(1.0 - sensitivity + 1e-12)
            log_probability_lesion_not_exists += math.log(specificity + 1e-12)

    logit = log_probability_lesion_exists - log_probability_lesion_not_exists
    posterior_prob = sigmoid_from_logit(logit)
    return float(np.clip(posterior_prob, 0.0, 1.0))


def run_m_step_standard(
    all_candidates: list[CandidateLesion],
    readers_by_image: dict[str, list],
    sensitivity_by_reader_and_class: dict,
    specificity_by_reader_and_class: dict,
) -> tuple[dict, dict, dict]:
    numerator_sensitivity   = {}
    denominator_sensitivity = {}
    numerator_specificity   = {}
    denominator_specificity = {}
    seen_count_by_reader_and_class = {}

    for candidate in all_candidates:
        all_readers_for_image = readers_by_image.get(candidate.image_id, [])
        supporting_reader_set = set(candidate.supporting_rad_ids)
        posterior             = candidate.posterior_prob

        for reader_id in all_readers_for_image:
            key = (reader_id, candidate.class_name)
            seen_count_by_reader_and_class[key] = seen_count_by_reader_and_class.get(key, 0) + 1

            if key not in numerator_sensitivity:
                numerator_sensitivity[key]   = EM_SMOOTHING_EPS
                denominator_sensitivity[key] = EM_SMOOTHING_EPS * 2
                numerator_specificity[key]   = EM_SMOOTHING_EPS
                denominator_specificity[key] = EM_SMOOTHING_EPS * 2

            denominator_sensitivity[key] += posterior
            denominator_specificity[key] += (1.0 - posterior)

            if reader_id in supporting_reader_set:
                numerator_sensitivity[key] += posterior
            else:
                numerator_specificity[key] += (1.0 - posterior)

    updated_sensitivity = {}
    updated_specificity = {}

    all_keys = set(sensitivity_by_reader_and_class.keys()) | set(numerator_sensitivity.keys())
    for key in all_keys:
        if key in numerator_sensitivity:
            raw_sensitivity = numerator_sensitivity[key] / max(denominator_sensitivity[key], 1e-12)
            raw_specificity = numerator_specificity[key] / max(denominator_specificity[key], 1e-12)
        else:
            raw_sensitivity = sensitivity_by_reader_and_class.get(key, INITIAL_SENSITIVITY)
            raw_specificity = specificity_by_reader_and_class.get(key, INITIAL_SPECIFICITY)

        updated_sensitivity[key] = float(np.clip(raw_sensitivity, SENSITIVITY_MIN, SENSITIVITY_MAX))
        updated_specificity[key] = float(np.clip(raw_specificity, SPECIFICITY_MIN, SPECIFICITY_MAX))

    return updated_sensitivity, updated_specificity, seen_count_by_reader_and_class

def run_m_step_shrinkage(
    all_candidates: list[CandidateLesion],
    readers_by_image: dict[str, list],
    sensitivity_by_reader_and_class: dict,
    specificity_by_reader_and_class: dict,
    reader_min_seen_for_learning: int,
    reader_shrinkage_strength: float,
) -> tuple[dict, dict, dict]:
    numerator_sensitivity = {}
    denominator_sensitivity = {}
    numerator_specificity = {}
    denominator_specificity = {}
    seen_count_by_reader_and_class = {}

    class_num_sensitivity = {}
    class_den_sensitivity = {}
    class_num_specificity = {}
    class_den_specificity = {}

    for candidate in all_candidates:
        all_readers_for_image = readers_by_image.get(candidate.image_id, [])
        supporting_reader_set = set(candidate.supporting_rad_ids)
        posterior = candidate.posterior_prob
        class_name = candidate.class_name

        for reader_id in all_readers_for_image:
            key = (reader_id, class_name)

            numerator_sensitivity.setdefault(key, 0.0)
            denominator_sensitivity.setdefault(key, 0.0)
            numerator_specificity.setdefault(key, 0.0)
            denominator_specificity.setdefault(key, 0.0)
            seen_count_by_reader_and_class.setdefault(key, 0)

            class_num_sensitivity.setdefault(class_name, 0.0)
            class_den_sensitivity.setdefault(class_name, 0.0)
            class_num_specificity.setdefault(class_name, 0.0)
            class_den_specificity.setdefault(class_name, 0.0)

            denominator_sensitivity[key] += posterior
            denominator_specificity[key] += 1.0 - posterior
            seen_count_by_reader_and_class[key] += 1

            class_den_sensitivity[class_name] += posterior
            class_den_specificity[class_name] += 1.0 - posterior

            if reader_id in supporting_reader_set:
                numerator_sensitivity[key] += posterior
                class_num_sensitivity[class_name] += posterior
            else:
                numerator_specificity[key] += 1.0 - posterior
                class_num_specificity[class_name] += 1.0 - posterior

    all_class_names = {class_name for _, class_name in sensitivity_by_reader_and_class.keys()}

    class_level_sensitivity = {}
    class_level_specificity = {}

    for class_name in all_class_names:
        class_level_sensitivity[class_name] = (
            class_num_sensitivity.get(class_name, 0.0)
            / max(class_den_sensitivity.get(class_name, 0.0), 1e-12)
        ) if class_den_sensitivity.get(class_name, 0.0) > 0 else INITIAL_SENSITIVITY

        class_level_specificity[class_name] = (
            class_num_specificity.get(class_name, 0.0)
            / max(class_den_specificity.get(class_name, 0.0), 1e-12)
        ) if class_den_specificity.get(class_name, 0.0) > 0 else INITIAL_SPECIFICITY

        class_level_sensitivity[class_name] = float(
            np.clip(class_level_sensitivity[class_name], SENSITIVITY_MIN, SENSITIVITY_MAX)
        )
        class_level_specificity[class_name] = float(
            np.clip(class_level_specificity[class_name], SPECIFICITY_MIN, SPECIFICITY_MAX)
        )

    updated_sensitivity = {}
    updated_specificity = {}

    all_keys = set(sensitivity_by_reader_and_class.keys()) | set(numerator_sensitivity.keys())

    for key in all_keys:
        _, class_name = key

        class_sens_prior = class_level_sensitivity.get(class_name, INITIAL_SENSITIVITY)
        class_spec_prior = class_level_specificity.get(class_name, INITIAL_SPECIFICITY)
        seen_count = seen_count_by_reader_and_class.get(key, 0)

        if seen_count < reader_min_seen_for_learning:
            updated_sensitivity[key] = class_sens_prior
            updated_specificity[key] = class_spec_prior
            continue

        sens_num = numerator_sensitivity.get(key, 0.0)
        sens_den = denominator_sensitivity.get(key, 0.0)
        spec_num = numerator_specificity.get(key, 0.0)
        spec_den = denominator_specificity.get(key, 0.0)

        shrunk_sensitivity = (
            sens_num + reader_shrinkage_strength * class_sens_prior
        ) / max(sens_den + reader_shrinkage_strength, 1e-12)

        shrunk_specificity = (
            spec_num + reader_shrinkage_strength * class_spec_prior
        ) / max(spec_den + reader_shrinkage_strength, 1e-12)

        updated_sensitivity[key] = float(
            np.clip(shrunk_sensitivity, SENSITIVITY_MIN, SENSITIVITY_MAX)
        )
        updated_specificity[key] = float(
            np.clip(shrunk_specificity, SPECIFICITY_MIN, SPECIFICITY_MAX)
        )

    return updated_sensitivity, updated_specificity, seen_count_by_reader_and_class


def run_candidate_em(
    all_candidates: list[CandidateLesion],
    readers_by_image: dict[str, list],
    sensitivity_by_reader_and_class: dict,
    specificity_by_reader_and_class: dict,
    num_em_iterations: int,
    prior_probability: float,
    em_strategy: str,
    reader_min_seen_for_learning: int,
    reader_shrinkage_strength: float,
) -> tuple[list[CandidateLesion], dict, dict, dict]:
    seen_count_by_reader_and_class = {}

    for iteration_index in range(num_em_iterations):
        for candidate in all_candidates:
            candidate.posterior_prob = run_e_step(
                candidate,
                readers_by_image,
                sensitivity_by_reader_and_class,
                specificity_by_reader_and_class,
                prior_probability=prior_probability,
            )

        if em_strategy == "standard_em":
            (
                sensitivity_by_reader_and_class,
                specificity_by_reader_and_class,
                seen_count_by_reader_and_class,
            ) = run_m_step_standard(
                all_candidates,
                readers_by_image,
                sensitivity_by_reader_and_class,
                specificity_by_reader_and_class,
            )

        elif em_strategy == "shrinkage_em":
            (
                sensitivity_by_reader_and_class,
                specificity_by_reader_and_class,
                seen_count_by_reader_and_class,
            ) = run_m_step_shrinkage(
                all_candidates,
                readers_by_image,
                sensitivity_by_reader_and_class,
                specificity_by_reader_and_class,
                reader_min_seen_for_learning,
                reader_shrinkage_strength,
            )

        else:
            raise ValueError(f"Unknown em_strategy: {em_strategy}")

    return (
        all_candidates,
        sensitivity_by_reader_and_class,
        specificity_by_reader_and_class,
        seen_count_by_reader_and_class,
    )

def aggregate_candidate_box(
    candidate_boxes: list[dict],
    sensitivity_by_reader_and_class: dict,
    localization_weight_by_reader_and_class: dict,
    class_name: str,
    image_width: float,
    image_height: float,
    coordinate_weighting: str = "sensitivity",
) -> tuple[float, float, float, float]:
    weights_list = []
    x_min_list   = []
    y_min_list   = []
    x_max_list   = []
    y_max_list   = []

    for box_record in candidate_boxes:
        reader_id   = box_record["rad_id"]
        key         = (reader_id, class_name)
        sensitivity = sensitivity_by_reader_and_class.get(key, INITIAL_SENSITIVITY)
        loc_weight  = localization_weight_by_reader_and_class.get(key, INITIAL_LOCALIZATION_WEIGHT)
        if coordinate_weighting == "sensitivity":
            weight = sensitivity * loc_weight
        elif coordinate_weighting == "uniform":
            weight = 1.0
        else:
            raise ValueError(f"Unknown coordinate_weighting: {coordinate_weighting}")

        weights_list.append(weight)
        x_min_list.append(box_record["x_min"])
        y_min_list.append(box_record["y_min"])
        x_max_list.append(box_record["x_max"])
        y_max_list.append(box_record["y_max"])

    total_weight = sum(weights_list)
    if total_weight <= 0.0:
        weights_list = [1.0] * len(candidate_boxes)
        total_weight = float(len(candidate_boxes))

    aggregated_x_min = sum(w * v for w, v in zip(weights_list, x_min_list)) / total_weight
    aggregated_y_min = sum(w * v for w, v in zip(weights_list, y_min_list)) / total_weight
    aggregated_x_max = sum(w * v for w, v in zip(weights_list, x_max_list)) / total_weight
    aggregated_y_max = sum(w * v for w, v in zip(weights_list, y_max_list)) / total_weight

    aggregated_x_min = float(np.clip(aggregated_x_min, 0.0, image_width))
    aggregated_y_min = float(np.clip(aggregated_y_min, 0.0, image_height))
    aggregated_x_max = float(np.clip(aggregated_x_max, 0.0, image_width))
    aggregated_y_max = float(np.clip(aggregated_y_max, 0.0, image_height))

    return aggregated_x_min, aggregated_y_min, aggregated_x_max, aggregated_y_max

def write_soft_consensus_csv(
    all_candidates: list[CandidateLesion],
    output_csv_path: Path,
    min_posterior: float,
    aggregation_method_label: str,
):
    output_rows = []
    for candidate in all_candidates:
        if candidate.posterior_prob < min_posterior:
            continue
        if candidate.class_id < 0:
            raise ValueError(f"Invalid class_id={candidate.class_id} for class {candidate.class_name}")
        output_rows.append({
            "image_id":           candidate.image_id,
            "class_name":         candidate.class_name,
            "class_id":           candidate.class_id,
            "x_min":              round(candidate.aggregated_box_xyxy[0], 4),
            "y_min":              round(candidate.aggregated_box_xyxy[1], 4),
            "x_max":              round(candidate.aggregated_box_xyxy[2], 4),
            "y_max":              round(candidate.aggregated_box_xyxy[3], 4),
            "posterior_prob":     round(candidate.posterior_prob, 4),
            "raw_support":        len(candidate.supporting_rad_ids),
            "num_readers":        candidate.num_readers,
            "supporting_rad_ids": "|".join(map(str, sorted(candidate.supporting_rad_ids, key=str))),
            "target_weight":      round(candidate.posterior_prob, 4),
            "aggregation_method": aggregation_method_label,
        })
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=SOFT_CONSENSUS_COLUMNS).to_csv(output_csv_path, index=False)
    log.info(f"Soft consensus CSV saved: {output_csv_path} ({len(output_rows)} rows)")


def write_reader_stats_csv(
    all_candidates: list[CandidateLesion],
    readers_by_image: dict[str, list],
    sensitivity_by_reader_and_class: dict,
    specificity_by_reader_and_class: dict,
    reader_stats_csv_path: Path,
    reader_min_seen_for_learning: int,
    em_strategy: str,
):
    reader_stats_rows = []
    all_keys = set(sensitivity_by_reader_and_class.keys())

    for (reader_id, class_name) in sorted(all_keys, key=lambda item: (str(item[0]), str(item[1]))):
        candidates_seen_by_reader = [
            candidate for candidate in all_candidates
            if candidate.class_name == class_name
            and reader_id in readers_by_image.get(candidate.image_id, [])
        ]
        candidates_supported_by_reader = [
            candidate for candidate in candidates_seen_by_reader
            if reader_id in candidate.supporting_rad_ids
        ]
        key = (reader_id, class_name)

        num_seen = len(candidates_seen_by_reader)
        is_reliable = num_seen >= reader_min_seen_for_learning

        if em_strategy == "shrinkage_em":
            parameter_source = (
                "reader_class_estimate"
                if is_reliable
                else "class_level_shrinkage_prior"
            )
        else:
            parameter_source = (
                "reader_class_estimate"
                if num_seen > 0
                else "initial_value"
            )

        reader_stats_rows.append({
            "rad_id": reader_id,
            "class_name": class_name,
            "sensitivity": round(sensitivity_by_reader_and_class.get(key, INITIAL_SENSITIVITY), 4),
            "candidate_specificity": round(specificity_by_reader_and_class.get(key, INITIAL_SPECIFICITY), 4),
            "num_supported_candidates": len(candidates_supported_by_reader),
            "num_seen_candidates": num_seen,
            "is_reliable_estimate": is_reliable,
            "parameter_source": parameter_source,
        })

    reader_stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reader_stats_rows, columns=READER_STATS_COLUMNS).to_csv(reader_stats_csv_path, index=False)
    log.info(f"Reader stats CSV saved: {reader_stats_csv_path}")


def write_bdc_raw_labels(
    local_boxes_dataframe: pd.DataFrame,
    class_config: dict,
    bdc_raw_dir: Path,
    all_image_ids: list,
):
    bdc_raw_dir.mkdir(parents=True, exist_ok=True)

    boxes_by_image = {
        image_id: image_boxes
        for image_id, image_boxes in local_boxes_dataframe.groupby("image_id")
    }

    number_of_files_written = 0
    number_of_empty_files = 0

    for image_id in all_image_ids:
        image_boxes = boxes_by_image.get(image_id)
        output_lines = []

        if image_boxes is not None:
            for _, box_row in image_boxes.iterrows():
                class_name = box_row["class_name"]
                class_id = class_config[class_name]["class_id"]

                if class_id < 0:
                    raise ValueError(f"Invalid class_id={class_id} for class {class_name}")

                output_lines.append(
                    f"{box_row['x_min']},{box_row['y_min']},{box_row['x_max']},{box_row['y_max']},"
                    f"{class_id},{box_row['rad_id']}"
                )

        output_file_path = bdc_raw_dir / f"{image_id}.txt"
        output_file_path.write_text("\n".join(output_lines))
        number_of_files_written += 1

        if not output_lines:
            number_of_empty_files += 1

    log.info(
        f"BDC raw labels written: {number_of_files_written} files, "
        f"{number_of_empty_files} empty -> {bdc_raw_dir}"
    )


def xyxy_to_yolo_xywh(
    x_min: float, y_min: float, x_max: float, y_max: float,
    image_width: float, image_height: float,
) -> tuple[float, float, float, float]:
    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width    = (x_max - x_min) / image_width
    height   = (y_max - y_min) / image_height

    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    width    = float(np.clip(width,    0.0, 1.0))
    height   = float(np.clip(height,   0.0, 1.0))

    return x_center, y_center, width, height

def candidate_box_tuple(candidate: CandidateLesion) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in candidate.aggregated_box_xyxy)


def candidates_match_for_postprocess(
    candidate_a: CandidateLesion,
    candidate_b: CandidateLesion,
    iou_threshold: float,
    use_center_distance: bool,
    center_distance_factor: float,
) -> bool:
    box_a = candidate_box_tuple(candidate_a)
    box_b = candidate_box_tuple(candidate_b)

    iou = compute_iou(box_a, box_b)
    if iou >= iou_threshold:
        return True

    if not use_center_distance:
        return False

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    if area_a <= 0.0 or area_b <= 0.0:
        return False

    mean_area = (area_a + area_b) / 2.0
    distance_limit = center_distance_factor * math.sqrt(mean_area)
    center_distance = compute_center_distance(box_a, box_b)

    return center_distance <= distance_limit


def fuse_candidate_group(candidates: list[CandidateLesion]) -> CandidateLesion:
    if len(candidates) == 1:
        return candidates[0]

    weights = np.array(
        [max(candidate.posterior_prob, 1e-6) for candidate in candidates],
        dtype=float,
    )
    weights = weights / weights.sum()

    boxes = np.array(
        [candidate_box_tuple(candidate) for candidate in candidates],
        dtype=float,
    )

    fused_box = tuple((boxes * weights[:, None]).sum(axis=0).tolist())

    first = candidates[0]

    supporting_rad_ids = sorted(
        set(
            rad_id
            for candidate in candidates
            for rad_id in candidate.supporting_rad_ids
        ),
        key=str,
    )

    merged_original_boxes = []
    for candidate in candidates:
        merged_original_boxes.extend(candidate.boxes)

    fused_candidate = CandidateLesion(
        image_id=first.image_id,
        class_name=first.class_name,
        class_id=first.class_id,
        boxes=merged_original_boxes,
        supporting_rad_ids=supporting_rad_ids,
        num_readers=max(candidate.num_readers for candidate in candidates),
        posterior_prob=max(candidate.posterior_prob for candidate in candidates),
        aggregated_box_xyxy=fused_box,
    )

    return fused_candidate


def postprocess_candidates_for_yolo_export(
    candidates: list[CandidateLesion],
    class_config: dict,
    strategy: str,
    default_iou_threshold: float,
    default_use_center_distance: bool,
    default_center_distance_factor: float,
) -> list[CandidateLesion]:
    if strategy == "none":
        return candidates

    candidates_by_image_and_class = {}

    for candidate in candidates:
        key = (candidate.image_id, candidate.class_name)
        candidates_by_image_and_class.setdefault(key, []).append(candidate)

    output_candidates = []

    for (image_id, class_name), group in candidates_by_image_and_class.items():
        cfg = class_config.get(class_name, {})

        iou_threshold = cfg.get("yolo_postprocess_iou", default_iou_threshold)
        use_center_distance = cfg.get(
            "yolo_postprocess_use_center_distance",
            default_use_center_distance,
        )
        center_distance_factor = cfg.get(
            "yolo_postprocess_center_distance_factor",
            default_center_distance_factor,
        )
        max_instances_per_image = cfg.get("max_instances_per_image", None)

        remaining = sorted(group, key=lambda c: c.posterior_prob, reverse=True)
        selected = []

        while remaining:
            seed = remaining.pop(0)
            matched = [seed]
            new_remaining = []

            for candidate in remaining:
                if candidates_match_for_postprocess(
                    seed,
                    candidate,
                    iou_threshold=iou_threshold,
                    use_center_distance=use_center_distance,
                    center_distance_factor=center_distance_factor,
                ):
                    matched.append(candidate)
                else:
                    new_remaining.append(candidate)

            if strategy == "same_class_nms":
                selected.append(seed)
            elif strategy == "same_class_wbf":
                selected.append(fuse_candidate_group(matched))
            else:
                raise ValueError(f"Unknown yolo_box_postprocess strategy: {strategy}")

            remaining = new_remaining

        selected = sorted(selected, key=lambda c: c.posterior_prob, reverse=True)

        if max_instances_per_image is not None:
            selected = selected[:int(max_instances_per_image)]

        output_candidates.extend(selected)

    return output_candidates


def write_yolo_labels(
    all_candidates: list[CandidateLesion],
    metadata_dataframe: pd.DataFrame,
    export_yolo_dir: Path,
    yolo_min_posterior: float,
    all_image_ids: list,
    class_config: dict | None = None,
    yolo_box_postprocess: str = "none",
    yolo_postprocess_iou: float = 0.45,
    yolo_postprocess_use_center_distance: bool = False,
    yolo_postprocess_center_distance_factor: float = 1.0,
):
    export_yolo_dir.mkdir(parents=True, exist_ok=True)

    image_dimensions_by_id = metadata_dataframe.set_index("image_id")[
        ["OriginalImage[Width]", "OriginalImage[Height]"]
    ].to_dict(orient="index")

    candidates_above_threshold = [
        candidate for candidate in all_candidates
        if candidate.posterior_prob >= yolo_min_posterior
    ]

    before_postprocess_count = len(candidates_above_threshold)

    if yolo_box_postprocess != "none":
        if class_config is None:
            raise ValueError("class_config is required when yolo_box_postprocess is enabled")

        candidates_above_threshold = postprocess_candidates_for_yolo_export(
            candidates=candidates_above_threshold,
            class_config=class_config,
            strategy=yolo_box_postprocess,
            default_iou_threshold=yolo_postprocess_iou,
            default_use_center_distance=yolo_postprocess_use_center_distance,
            default_center_distance_factor=yolo_postprocess_center_distance_factor,
        )

        log.info(
            f"YOLO label postprocess '{yolo_box_postprocess}': "
            f"{before_postprocess_count} -> {len(candidates_above_threshold)} candidates"
    )

    candidates_by_image: dict[str, list[CandidateLesion]] = {image_id: [] for image_id in all_image_ids}
    for candidate in candidates_above_threshold:
        candidates_by_image.setdefault(candidate.image_id, []).append(candidate)

    number_of_empty_label_files = 0
    number_of_label_files       = 0

    for image_id in all_image_ids:
        image_candidates = candidates_by_image.get(image_id, [])
        label_lines      = []

        if image_id not in image_dimensions_by_id:
            raise ValueError(f"No metadata for image_id={image_id}, cannot write YOLO label")

        image_width  = image_dimensions_by_id[image_id]["OriginalImage[Width]"]
        image_height = image_dimensions_by_id[image_id]["OriginalImage[Height]"]

        for candidate in image_candidates:
            if candidate.class_id < 0:
                raise ValueError(f"Invalid class_id={candidate.class_id} for class {candidate.class_name}")

            x_min, y_min, x_max, y_max = candidate.aggregated_box_xyxy
            x_center, y_center, width, height = xyxy_to_yolo_xywh(
                x_min, y_min, x_max, y_max, image_width, image_height
            )
            if width <= 0.0 or height <= 0.0:
                continue
            label_lines.append(
                f"{candidate.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        output_label_path = export_yolo_dir / f"{image_id}.txt"
        output_label_path.write_text("\n".join(label_lines))
        number_of_label_files += 1

        if not label_lines:
            number_of_empty_label_files += 1

    log.info(f"YOLO label files written: {number_of_label_files} total, {number_of_empty_label_files} empty")

def run_tests():
    log.info("Running built-in tests...")

    dummy_sensitivity = {("R1", "Nodule/Mass"): 0.75, ("R2", "Nodule/Mass"): 0.75, ("R3", "Nodule/Mass"): 0.75}
    dummy_specificity = {("R1", "Nodule/Mass"): 0.90, ("R2", "Nodule/Mass"): 0.90, ("R3", "Nodule/Mass"): 0.90}
    readers_by_image_test = {"img1": ["R1", "R2", "R3"]}

    candidate_one_supporter = CandidateLesion(
        image_id="img1", class_name="Nodule/Mass", class_id=3,
        boxes=[], supporting_rad_ids=["R1"], num_readers=3, posterior_prob=0.5,
    )
    posterior_one = run_e_step(candidate_one_supporter, readers_by_image_test, dummy_sensitivity, dummy_specificity)

    candidate_three_supporters = CandidateLesion(
        image_id="img1", class_name="Nodule/Mass", class_id=3,
        boxes=[], supporting_rad_ids=["R1", "R2", "R3"], num_readers=3, posterior_prob=0.5,
    )
    posterior_three = run_e_step(candidate_three_supporters, readers_by_image_test, dummy_sensitivity, dummy_specificity)

    assert posterior_one < posterior_three, "EM posterior test failed: single supporter should have lower posterior"
    log.info("EM posterior test passed: single supporter < three supporters posterior")

    # Test A: YOLO conversion
    x_center, y_center, width, height = xyxy_to_yolo_xywh(400.0, 100.0, 600.0, 200.0, 1000.0, 500.0)
    assert abs(x_center - 0.5)  < 1e-6, f"Test A x_center failed: {x_center}"
    assert abs(y_center - 0.3)  < 1e-6, f"Test A y_center failed: {y_center}"
    assert abs(width    - 0.2)  < 1e-6, f"Test A width failed: {width}"
    assert abs(height   - 0.2)  < 1e-6, f"Test A height failed: {height}"
    log.info("Test A passed: YOLO xywh conversion correct")

    # Test B: two distant boxes from the same radiologist remain two candidates.
    test_boxes_distant = [
        {"rad_id": "R1", "x_min_norm": 0.1, "y_min_norm": 0.1, "x_max_norm": 0.2, "y_max_norm": 0.2,
         "x_min": 100.0, "y_min": 100.0, "x_max": 200.0, "y_max": 200.0},
        {"rad_id": "R1", "x_min_norm": 0.7, "y_min_norm": 0.7, "x_max_norm": 0.9, "y_max_norm": 0.9,
         "x_min": 700.0, "y_min": 700.0, "x_max": 900.0, "y_max": 900.0},
    ]
    clusters_test_b = build_candidate_clusters(
        test_boxes_distant,
        iou_threshold=0.35,
        center_distance_factor=1.5,
        use_center_distance=False,
    )
    assert len(clusters_test_b) == 2, f"Test B failed: expected 2 clusters, got {len(clusters_test_b)}"
    log.info("Test B passed: two distant boxes from same reader remain separate")

    # Test C: near-identical boxes from the same radiologist are deduplicated.
    test_boxes_near_duplicates = [
        {"rad_id": "R1", "x_min_norm": 0.40, "y_min_norm": 0.40, "x_max_norm": 0.60, "y_max_norm": 0.60,
         "x_min": 400.0, "y_min": 400.0, "x_max": 600.0, "y_max": 600.0},
        {"rad_id": "R1", "x_min_norm": 0.405, "y_min_norm": 0.405, "x_max_norm": 0.605, "y_max_norm": 0.605,
         "x_min": 405.0, "y_min": 405.0, "x_max": 605.0, "y_max": 605.0},
    ]
    deduplicated_test_c = deduplicate_within_reader(test_boxes_near_duplicates)
    assert len(deduplicated_test_c) == 1, f"Test C failed: expected 1 box after dedup, got {len(deduplicated_test_c)}"
    log.info("Test C passed: near-duplicate boxes from same reader merged")

    # Test D: constrained clustering never puts two boxes from the same radiologist in one cluster.
    test_boxes_connected_component_trap = [
        {"rad_id": "R1", "x_min_norm": 0.10, "y_min_norm": 0.10, "x_max_norm": 0.30, "y_max_norm": 0.30,
         "x_min": 100.0, "y_min": 100.0, "x_max": 300.0, "y_max": 300.0},
        {"rad_id": "R2", "x_min_norm": 0.12, "y_min_norm": 0.12, "x_max_norm": 0.32, "y_max_norm": 0.32,
         "x_min": 120.0, "y_min": 120.0, "x_max": 320.0, "y_max": 320.0},
        {"rad_id": "R1", "x_min_norm": 0.14, "y_min_norm": 0.14, "x_max_norm": 0.34, "y_max_norm": 0.34,
         "x_min": 140.0, "y_min": 140.0, "x_max": 340.0, "y_max": 340.0},
    ]
    clusters_test_d = build_candidate_clusters(
        test_boxes_connected_component_trap,
        iou_threshold=0.20,
        center_distance_factor=1.5,
        use_center_distance=False,
    )
    for cluster in clusters_test_d:
        assert len({b["rad_id"] for b in cluster}) == len(cluster), "Test D failed: duplicate rad_id in a cluster"
    log.info("Test D passed: constrained clusters have unique rad_id values")

    # Test E: one No finding image receives empty YOLO and BDC raw export files.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        yolo_dir = tmp_path / "yolo"
        bdc_dir = tmp_path / "bdc"
        all_image_ids = ["normal_img"]
        metadata_dataframe = pd.DataFrame({
            "image_id": ["normal_img"],
            "OriginalImage[Width]": [1000],
            "OriginalImage[Height]": [500],
        })
        empty_local_boxes = pd.DataFrame(columns=REQUIRED_ANNOTATION_COLUMNS)
        write_yolo_labels([], metadata_dataframe, yolo_dir, 0.5, all_image_ids)
        write_bdc_raw_labels(empty_local_boxes, {"Nodule/Mass": {"class_id": 3}}, bdc_dir, all_image_ids)

        yolo_file = yolo_dir / "normal_img.txt"
        bdc_file = bdc_dir / "normal_img.txt"
        assert yolo_file.exists(), "Test E failed: missing empty YOLO label file"
        assert bdc_file.exists(), "Test E failed: missing empty BDC raw label file"
        assert yolo_file.read_text() == "", "Test E failed: YOLO label file should be empty"
        assert bdc_file.read_text() == "", "Test E failed: BDC raw label file should be empty"
    log.info("Test E passed: normal image receives empty YOLO and BDC files")

    # Test F: missing class configuration raises ValueError.
    local_boxes_missing_class = pd.DataFrame({
        "image_id": ["img1"],
        "rad_id": ["R1"],
        "class_name": ["Missing Class"],
        "x_min": [1],
        "y_min": [1],
        "x_max": [2],
        "y_max": [2],
    })
    try:
        validate_class_config_covers_observed_classes(local_boxes_missing_class, {"Nodule/Mass": {"class_id": 3}})
    except ValueError:
        pass
    else:
        raise AssertionError("Test F failed: expected ValueError for missing class config")
    log.info("Test F passed: missing class config raises ValueError")

    # Test G: standard EM m-step returns seen counts.
    standard_sens, standard_spec, standard_seen = run_m_step_standard(
        [candidate_one_supporter, candidate_three_supporters],
        readers_by_image_test,
        dummy_sensitivity,
        dummy_specificity,
    )

    assert isinstance(standard_seen, dict), "Test G failed: standard EM should return seen count dict"
    log.info("Test G passed: standard EM returns seen counts")


    # Test H: shrinkage EM runs and returns valid parameters.
    shrink_sens, shrink_spec, shrink_seen = run_m_step_shrinkage(
        [candidate_one_supporter, candidate_three_supporters],
        readers_by_image_test,
        dummy_sensitivity,
        dummy_specificity,
        reader_min_seen_for_learning=30,
        reader_shrinkage_strength=50.0,
    )

    for value in shrink_sens.values():
        assert SENSITIVITY_MIN <= value <= SENSITIVITY_MAX

    for value in shrink_spec.values():
        assert SPECIFICITY_MIN <= value <= SPECIFICITY_MAX

    assert isinstance(shrink_seen, dict), "Test H failed: shrinkage EM should return seen count dict"
    log.info("Test H passed: shrinkage EM returns valid clipped parameters")

    # Test I: coordinate weighting modes both run.
    dummy_box_records = [
        {
            "rad_id": "R1",
            "x_min": 100.0,
            "y_min": 100.0,
            "x_max": 200.0,
            "y_max": 200.0,
        },
        {
            "rad_id": "R2",
            "x_min": 120.0,
            "y_min": 120.0,
            "x_max": 220.0,
            "y_max": 220.0,
        },
    ]

    uniform_box = aggregate_candidate_box(
        dummy_box_records,
        dummy_sensitivity,
        {},
        "Nodule/Mass",
        1000.0,
        1000.0,
        coordinate_weighting="uniform",
    )

    sensitivity_box = aggregate_candidate_box(
        dummy_box_records,
        dummy_sensitivity,
        {},
        "Nodule/Mass",
        1000.0,
        1000.0,
        coordinate_weighting="sensitivity",
    )

    assert len(uniform_box) == 4
    assert len(sensitivity_box) == 4
    log.info("Test I passed: coordinate weighting modes run")

    # Test J: YOLO postprocess merges overlapping same-class candidates.
    candidate_a = CandidateLesion(
        image_id="img1",
        class_name="Cardiomegaly",
        class_id=3,
        boxes=[],
        supporting_rad_ids=["R1"],
        num_readers=3,
        posterior_prob=0.9,
        aggregated_box_xyxy=(100.0, 100.0, 300.0, 300.0),
    )

    candidate_b = CandidateLesion(
        image_id="img1",
        class_name="Cardiomegaly",
        class_id=3,
        boxes=[],
        supporting_rad_ids=["R2"],
        num_readers=3,
        posterior_prob=0.8,
        aggregated_box_xyxy=(110.0, 110.0, 310.0, 310.0),
    )

    postprocessed = postprocess_candidates_for_yolo_export(
        candidates=[candidate_a, candidate_b],
        class_config={
            "Cardiomegaly": {
                "class_id": 3,
                "yolo_postprocess_iou": 0.3,
                "yolo_postprocess_use_center_distance": True,
                "yolo_postprocess_center_distance_factor": 1.2,
                "max_instances_per_image": 1,
            }
        },
        strategy="same_class_wbf",
        default_iou_threshold=0.45,
        default_use_center_distance=False,
        default_center_distance_factor=1.0,
    )

    assert len(postprocessed) == 1, "Test J failed: overlapping same-class candidates should be merged"
    log.info("Test J passed: YOLO postprocess merges overlapping same-class candidates")

    # Extra test where center-distance fallback is disabled unless requested.
    box_a = (0.10, 0.10, 0.20, 0.20)
    box_b = (0.21, 0.10, 0.31, 0.20)
    assert not boxes_match(box_a, box_b, iou_threshold=0.35, center_distance_factor=2.0, use_center_distance=False)
    assert boxes_match(box_a, box_b, iou_threshold=0.35, center_distance_factor=2.0, use_center_distance=True)
    log.info("Center-distance gating test passed")

    log.info("All tests passed.")

def main():
    parser = argparse.ArgumentParser(
        description="Offline candidate-level EM / BDC-inspired consensus preprocessing for VinDr-CXR"
    )
    parser.add_argument("--input_csv",          type=Path)
    parser.add_argument("--metadata_csv",       type=Path)
    parser.add_argument("--output_csv",         type=Path)
    parser.add_argument("--reader_stats_csv",   type=Path)
    parser.add_argument("--bdc_raw_dir",        type=Path)
    parser.add_argument("--export_yolo_dir",    type=Path)
    parser.add_argument("--class_config_json",  type=Path)
    parser.add_argument("--num_em_iterations",  default=40,     type=int)
    parser.add_argument("--prior_probability",  default=0.5,    type=float)
    parser.add_argument("--min_posterior",      default=0.25,   type=float)
    parser.add_argument("--yolo_min_posterior", default=0.4,    type=float)
    parser.add_argument("--run_tests",          action="store_true")
    parser.add_argument(
        "--allow_class_subset",
        action="store_true",
        help="If set, ignore annotation classes missing from class_config_json instead of raising an error.",
    )
    parser.add_argument(
        "--em_strategy",
        choices=["standard_em", "shrinkage_em"],
        default="standard_em",
        help="standard_em = original EM or shrinkage_em = stabilized EM with class-level shrinkage.",
    )

    parser.add_argument(
        "--reader_min_seen_for_learning",
        default=30,
        type=int,
        help="Minimum candidate count for trusting reader-class reliability in shrinkage_em.",
    )

    parser.add_argument(
        "--reader_shrinkage_strength",
        default=50.0,
        type=float,
        help="Pseudo-count strength for shrinkage_em.",
    )

    parser.add_argument(
        "--coordinate_weighting",
        choices=["uniform", "sensitivity"],
        default="sensitivity",
        help="sensitivity = old behavior; uniform = safer with uneven reader distribution.",
    )
    parser.add_argument(
        "--yolo_box_postprocess",
        choices=["none", "same_class_nms", "same_class_wbf"],
        default="none",
        help="Optional same-class box deduplication/fusion before YOLO export.",
    )
    parser.add_argument(
        "--yolo_postprocess_iou",
        default=0.45,
        type=float,
        help="IoU threshold for same-class YOLO label postprocessing.",
    )
    parser.add_argument(
        "--yolo_postprocess_use_center_distance",
        action="store_true",
        help="Also merge boxes with nearby centers, useful for CXR boxes.",
    )
    parser.add_argument(
        "--yolo_postprocess_center_distance_factor",
        default=1.0,
        type=float,
    )
    args = parser.parse_args()

    if args.run_tests:
        run_tests()
        return

    required_args = [
        "input_csv",
        "metadata_csv",
        "output_csv",
        "reader_stats_csv",
        "bdc_raw_dir",
        "export_yolo_dir",
        "class_config_json",
    ]

    missing_args = [arg for arg in required_args if getattr(args, arg) is None]

    if missing_args:
        parser.error(f"Missing required arguments: {missing_args}")

    if not (0.0 < args.prior_probability < 1.0):
        raise ValueError("--prior_probability must be between 0 and 1")

    annotations_dataframe, metadata_dataframe = load_inputs(args.input_csv, args.metadata_csv)
    validate_metadata(metadata_dataframe)
    class_config = load_class_config(args.class_config_json)

    log.info(f"Images:                 {annotations_dataframe['image_id'].nunique()}")
    log.info(f"Radiologists:           {annotations_dataframe['rad_id'].nunique()}")
    log.info(f"Total annotation rows:  {len(annotations_dataframe)}")

    no_finding_count = (annotations_dataframe["class_name"] == NO_FINDING_LABEL).sum()
    log.info(f"No finding rows:        {no_finding_count}")

    readers_by_image = build_readers_by_image(annotations_dataframe)
    all_image_ids = sorted(readers_by_image.keys(), key=str)

    local_boxes_df = filter_local_boxes(annotations_dataframe)

    if args.allow_class_subset:
        local_boxes_df = filter_boxes_to_configured_classes(local_boxes_df, class_config)
    else:
        validate_class_config_covers_observed_classes(local_boxes_df, class_config)

    local_boxes_df = attach_image_dimensions(local_boxes_df, metadata_dataframe)
    local_boxes_df = clean_and_validate_box_coordinates(local_boxes_df)
    local_boxes_df = normalize_xyxy_boxes(local_boxes_df)

    log.info(f"Local box rows:         {len(local_boxes_df)}")

    image_dimensions_by_id = metadata_dataframe.set_index("image_id")[
        ["OriginalImage[Width]", "OriginalImage[Height]"]
    ].to_dict(orient="index")

    all_reader_ids  = sorted(annotations_dataframe["rad_id"].unique().tolist(), key=str)
    all_class_names = sorted(local_boxes_df["class_name"].unique().tolist(), key=str)

    sensitivity_by_reader_and_class, specificity_by_reader_and_class, localization_weight_by_reader_and_class = (
        initialize_reader_parameters(all_reader_ids, all_class_names)
    )

    all_candidates: list[CandidateLesion] = []

    for image_id, image_annotations in local_boxes_df.groupby("image_id"):
        if image_id not in image_dimensions_by_id:
            raise ValueError(f"No metadata for image_id={image_id}, cannot aggregate boxes")

        num_readers_for_image = len(readers_by_image.get(image_id, []))
        image_width  = image_dimensions_by_id[image_id]["OriginalImage[Width]"]
        image_height = image_dimensions_by_id[image_id]["OriginalImage[Height]"]

        for class_name, class_annotations in image_annotations.groupby("class_name"):
            cfg = class_config[class_name]
            iou_threshold          = cfg.get("iou_threshold",          DEFAULT_IOU_THRESHOLD)
            center_distance_factor = cfg.get("center_distance_factor", DEFAULT_CENTER_DISTANCE_FACTOR)
            use_center_distance    = cfg.get("use_center_distance",    False)
            class_id               = cfg["class_id"]

            reader_deduplicated_boxes: list[dict] = []
            for _, reader_boxes_df in class_annotations.groupby("rad_id"):
                reader_box_records = reader_boxes_df.to_dict(orient="records")
                reader_deduplicated_boxes.extend(deduplicate_within_reader(reader_box_records))

            clusters = build_candidate_clusters(
                reader_deduplicated_boxes,
                iou_threshold,
                center_distance_factor,
                use_center_distance,
            )

            for cluster_boxes in clusters:
                supporting_rad_ids = sorted(set(box["rad_id"] for box in cluster_boxes), key=str)
                raw_support        = len(supporting_rad_ids)
                initial_posterior  = raw_support / max(num_readers_for_image, 1)

                candidate = CandidateLesion(
                    image_id=image_id,
                    class_name=class_name,
                    class_id=class_id,
                    boxes=cluster_boxes,
                    supporting_rad_ids=supporting_rad_ids,
                    num_readers=num_readers_for_image,
                    posterior_prob=initial_posterior,
                )
                all_candidates.append(candidate)

    log.info(f"Candidates before EM:   {len(all_candidates)}")

    (
        all_candidates,
        sensitivity_by_reader_and_class,
        specificity_by_reader_and_class,
        seen_count_by_reader_and_class,
    ) = run_candidate_em(
        all_candidates,
        readers_by_image,
        sensitivity_by_reader_and_class,
        specificity_by_reader_and_class,
        args.num_em_iterations,
        args.prior_probability,
        args.em_strategy,
        args.reader_min_seen_for_learning,
        args.reader_shrinkage_strength,
    )

    for candidate in all_candidates:
        if candidate.image_id not in image_dimensions_by_id:
            raise ValueError(f"No metadata for image_id={candidate.image_id}, cannot aggregate boxes")

        image_width  = image_dimensions_by_id[candidate.image_id]["OriginalImage[Width]"]
        image_height = image_dimensions_by_id[candidate.image_id]["OriginalImage[Height]"]

        candidate.aggregated_box_xyxy = aggregate_candidate_box(
            candidate.boxes,
            sensitivity_by_reader_and_class,
            localization_weight_by_reader_and_class,
            candidate.class_name,
            image_width,
            image_height,
            coordinate_weighting=args.coordinate_weighting,
        )

    candidates_after_filter = [c for c in all_candidates if c.posterior_prob >= args.min_posterior]
    log.info(f"Candidates after filter: {len(candidates_after_filter)}")

    aggregation_method_label = (
        f"candidate_level_{args.em_strategy}_"
        f"{args.coordinate_weighting}_coordinate_weighting"
    )

    write_soft_consensus_csv(
        all_candidates,
        args.output_csv,
        args.min_posterior,
        aggregation_method_label,
    )
    write_reader_stats_csv(
        all_candidates,
        readers_by_image,
        sensitivity_by_reader_and_class,
        specificity_by_reader_and_class,
        args.reader_stats_csv,
        args.reader_min_seen_for_learning,
        args.em_strategy,
    )
    write_bdc_raw_labels(
        local_boxes_df,
        class_config,
        args.bdc_raw_dir,
        all_image_ids,
    )
    write_yolo_labels(
        all_candidates,
        metadata_dataframe,
        args.export_yolo_dir,
        args.yolo_min_posterior,
        all_image_ids,
        class_config=class_config,
        yolo_box_postprocess=args.yolo_box_postprocess,
        yolo_postprocess_iou=args.yolo_postprocess_iou,
        yolo_postprocess_use_center_distance=args.yolo_postprocess_use_center_distance,
        yolo_postprocess_center_distance_factor=args.yolo_postprocess_center_distance_factor,
    )

if __name__ == "__main__":
    main()
