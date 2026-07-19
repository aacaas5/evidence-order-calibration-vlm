import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


SEED = 42
GRID_STEPS = 41
MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_accepted.json")
SCENES = Path("data/gqa/metadata/val_sceneGraphs.json")
IMAGE_DIR = Path("data/gqa/scaled_images")
OUT_DIR = Path("results/scaled/p18")
BOXES_PATH = OUT_DIR / "control_boxes.json"
SUMMARY_PATH = OUT_DIR / "control_box_summary.json"
QC_PATH = OUT_DIR / "control_box_examples.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(box_a, box_b):
    return max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * max(
        0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    )


def iou(box_a, box_b):
    overlap = intersection(box_a, box_b)
    union = area(box_a) + area(box_b) - overlap
    return overlap / union if union > 0 else 0.0


def object_boxes(scene, critical_object_id, image_width, image_height):
    boxes = []
    for object_id, obj in scene.get("objects", {}).items():
        if str(object_id) == str(critical_object_id):
            continue
        try:
            x, y = float(obj["x"]), float(obj["y"])
            width, height = float(obj["w"]), float(obj["h"])
        except (KeyError, TypeError, ValueError):
            continue
        box = [
            max(0.0, x),
            max(0.0, y),
            min(float(image_width), x + width),
            min(float(image_height), y + height),
        ]
        if area(box) > 0:
            boxes.append(box)
    return boxes


def select_control(sample, scene):
    critical = list(map(float, sample["critical_objects"][0]["bbox_xyxy"]))
    image_width = float(sample["image_width"])
    image_height = float(sample["image_height"])
    box_width = critical[2] - critical[0]
    box_height = critical[3] - critical[1]
    if box_width <= 0 or box_height <= 0 or box_width > image_width or box_height > image_height:
        return None

    annotations = object_boxes(
        scene,
        sample["critical_objects"][0]["object_id"],
        image_width,
        image_height,
    )
    x_positions = np.linspace(0.0, image_width - box_width, GRID_STEPS)
    y_positions = np.linspace(0.0, image_height - box_height, GRID_STEPS)
    critical_center = ((critical[0] + critical[2]) / 2, (critical[1] + critical[3]) / 2)
    candidates = []
    control_area = box_width * box_height
    for y1 in y_positions:
        for x1 in x_positions:
            candidate = [float(x1), float(y1), float(x1 + box_width), float(y1 + box_height)]
            critical_iou = iou(candidate, critical)
            if critical_iou > 0.05 + 1e-12:
                continue
            overlaps = [intersection(candidate, annotated) / control_area for annotated in annotations]
            overlap_score = float(sum(overlaps))
            overlapping_objects = int(sum(value > 0 for value in overlaps))
            center = ((candidate[0] + candidate[2]) / 2, (candidate[1] + candidate[3]) / 2)
            distance_squared = (center[0] - critical_center[0]) ** 2 + (center[1] - critical_center[1]) ** 2
            sort_key = (
                round(overlap_score, 12),
                -round(distance_squared, 12),
                round(candidate[0], 12),
                round(candidate[1], 12),
                round(candidate[2], 12),
                round(candidate[3], 12),
            )
            candidates.append((sort_key, candidate, overlap_score, critical_iou, overlapping_objects))
    if not candidates:
        return None
    _, candidate, overlap_score, critical_iou, overlapping_objects = min(candidates, key=lambda item: item[0])
    return {
        "control_bbox": [round(value, 6) for value in candidate],
        "control_overlap_score": overlap_score,
        "critical_control_iou": critical_iou,
        "overlapping_noncritical_objects": overlapping_objects,
        "annotated_noncritical_objects": len(annotations),
    }


samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
scenes = json.loads(SCENES.read_text(encoding="utf-8"))
rows = []
for index, sample in enumerate(samples, 1):
    image_id = str(sample["image_id"])
    scene = scenes.get(image_id)
    result = select_control(sample, scene) if scene else None
    row = {
        "question_id": str(sample["question_id"]),
        "image_id": image_id,
        "critical_bbox": sample["critical_objects"][0]["bbox_xyxy"],
        "control_bbox": result["control_bbox"] if result else None,
        "control_overlap_score": result["control_overlap_score"] if result else None,
        "control_valid": result is not None,
    }
    if result:
        row.update({key: value for key, value in result.items() if key != "control_bbox"})
    else:
        row["invalid_reason"] = "Missing scene graph or no same-size in-bounds candidate satisfying IoU <= 0.05"
    rows.append(row)
    print(
        f"[{index}/{len(samples)}] qid={row['question_id']} valid={row['control_valid']} "
        f"overlap={row['control_overlap_score']}"
    )

BOXES_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
valid = [row for row in rows if row["control_valid"]]
summary = {
    "seed": SEED,
    "selection_uses_randomness": False,
    "grid_steps_per_axis": GRID_STEPS,
    "overlap_score_definition": "sum of intersection_area/control_box_area over all non-critical scene-graph objects",
    "total_samples": len(rows),
    "valid_matched_controls": len(valid),
    "invalid_controls": len(rows) - len(valid),
    "mean_control_overlap_score": float(np.mean([row["control_overlap_score"] for row in valid])) if valid else None,
    "zero_overlap_controls": int(sum(row["control_overlap_score"] == 0 for row in valid)),
    "mean_critical_control_iou": float(np.mean([row["critical_control_iou"] for row in valid])) if valid else None,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Deterministic evenly spaced visual QC subset.
if valid:
    qc_indices = np.linspace(0, len(valid) - 1, min(12, len(valid)), dtype=int)
    figure, axes = plt.subplots(3, 4, figsize=(16, 11))
    for axis, valid_index in zip(axes.flat, qc_indices):
        row = valid[int(valid_index)]
        image = Image.open(IMAGE_DIR / f'{row["image_id"]}.jpg').convert("RGB")
        axis.imshow(image)
        critical = row["critical_bbox"]
        control = row["control_bbox"]
        axis.add_patch(Rectangle(
            (critical[0], critical[1]), critical[2] - critical[0], critical[3] - critical[1],
            fill=False, edgecolor="red", linewidth=2.2, label="critical",
        ))
        axis.add_patch(Rectangle(
            (control[0], control[1]), control[2] - control[0], control[3] - control[1],
            fill=False, edgecolor="cyan", linewidth=2.2, label="irrelevant",
        ))
        axis.set_title(
            f"QID {row['question_id']}\noverlap={row['control_overlap_score']:.3f}", fontsize=9
        )
        axis.axis("off")
    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2.2, label="critical"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="cyan", linewidth=2.2, label="matched irrelevant"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.suptitle("P18 deterministic matched-control QC", fontsize=14)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(QC_PATH, dpi=180)
    plt.close(figure)

print("P18 CONTROL BOX CONSTRUCTION")
print("Total samples:", summary["total_samples"])
print("Valid matched controls:", summary["valid_matched_controls"])
print("Invalid controls:", summary["invalid_controls"])
print("Mean control overlap score:", summary["mean_control_overlap_score"])
print("Control boxes:", BOXES_PATH)
print("Summary:", SUMMARY_PATH)
print("QC figure:", QC_PATH)
