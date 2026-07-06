import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_accepted.json")
IMAGE_DIR = Path("data/gqa/scaled_images")
P5_DIR = Path("results/scaled/p5")
P6_DIR = Path("results/scaled/p6")
RESULT_PATH = P5_DIR / "results.json"
SUMMARY_PATH = P5_DIR / "summary.json"
PROGRESS_PATH = P6_DIR / "hidden_progress.npz"
FEATURES_PATH = P6_DIR / "hidden_features.npz"
META_PATH = P6_DIR / "hidden_features_meta.json"
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]

P5_DIR.mkdir(parents=True, exist_ok=True)
P6_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text):
    text = re.sub(r"[^\w\s-]", "", str(text).lower().strip())
    return " ".join(text.split())


def corrupt(image, box, severity):
    if severity == 0:
        return image.copy()
    x1, y1, x2, y2 = map(float, box)
    scale = math.sqrt(severity)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    output = image.copy()
    ImageDraw.Draw(output).rectangle(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        fill=(127, 127, 127),
    )
    return output


def key(question_id, severity):
    return f"{question_id}|{severity:.2f}"


def save_results(rows):
    ordered = sorted(rows.values(), key=lambda row: (row["question_id"], row["severity"]))
    RESULT_PATH.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def save_hidden(vectors):
    keys = sorted(vectors)
    matrix = np.stack([vectors[item] for item in keys]).astype(np.float32)
    np.savez_compressed(PROGRESS_PATH, keys=np.asarray(keys), X=matrix)


def summarize(rows, samples):
    ordered = list(rows.values())
    severity_summary = {}
    for severity in SEVERITIES:
        subset = [row for row in ordered if float(row["severity"]) == severity]
        if subset:
            severity_summary[str(severity)] = {
                "n": len(subset),
                "accuracy": float(np.mean([row["correct"] for row in subset])),
                "mean_c_seq": float(np.mean([row["c_seq"] for row in subset])),
                "mean_entropy": float(np.mean([row["entropy"] for row in subset])),
            }

    trajectories = defaultdict(list)
    for row in ordered:
        trajectories[row["question_id"]].append(row)

    def violation_metrics(selected):
        violations = pairs = trajectory_violations = complete = 0
        by_category = defaultdict(lambda: [0, 0])
        for question_id, trajectory in selected.items():
            trajectory = sorted(trajectory, key=lambda row: row["severity"])
            if len(trajectory) != len(SEVERITIES):
                continue
            complete += 1
            has_violation = False
            for left, right in zip(trajectory[:-1], trajectory[1:]):
                violation = right["c_seq"] > left["c_seq"]
                violations += int(violation)
                pairs += 1
                category = left["category"]
                by_category[category][0] += int(violation)
                by_category[category][1] += 1
                has_violation |= violation
            trajectory_violations += int(has_violation)
        return {
            "complete_trajectories": complete,
            "adjacent_pairs": pairs,
            "adjacent_violations": violations,
            "emvr": violations / pairs if pairs else None,
            "trajectories_with_violation": trajectory_violations,
            "trajectory_violation_rate": trajectory_violations / complete if complete else None,
            "category_emvr": {
                category: {"violations": values[0], "pairs": values[1], "emvr": values[0] / values[1]}
                for category, values in sorted(by_category.items())
            },
        }

    all_metrics = violation_metrics(trajectories)
    clean_correct_ids = {
        question_id
        for question_id, trajectory in trajectories.items()
        if any(row["severity"] == 0.0 and row["correct"] for row in trajectory)
    }
    clean_metrics = violation_metrics(
        {question_id: trajectories[question_id] for question_id in clean_correct_ids}
    )
    return {
        "accepted_samples": len(samples),
        "conditions": len(ordered),
        "planned_conditions": len(samples) * len(SEVERITIES),
        "severity_summary": severity_summary,
        **all_metrics,
        "clean_correct_trajectories": len(clean_correct_ids),
        "clean_correct_emvr": clean_metrics["emvr"],
        "clean_correct_trajectory_violation_rate": clean_metrics["trajectory_violation_rate"],
        "clean_correct_details": clean_metrics,
    }


samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
rows = {}
if RESULT_PATH.exists():
    for row in json.loads(RESULT_PATH.read_text(encoding="utf-8")):
        rows[key(str(row["question_id"]), float(row["severity"]))] = row

vectors = {}
if PROGRESS_PATH.exists():
    progress = np.load(PROGRESS_PATH)
    vectors = {str(item): vector for item, vector in zip(progress["keys"], progress["X"])}

print("P10/P11 SCALED QWEN")
print("Accepted trajectories:", len(samples))
print("Planned conditions:", len(samples) * len(SEVERITIES))
print("Cached result rows:", len(rows))
print("Cached hidden vectors:", len(vectors))

processor = AutoProcessor.from_pretrained(MODEL)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype="auto", device_map="auto"
)
model.eval()

layer_candidates = []
for module_name, module in model.named_modules():
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[-2] == "layers":
        try:
            layer_candidates.append((int(parts[-1]), module_name, module))
        except ValueError:
            pass
if not layer_candidates:
    raise RuntimeError("Could not locate transformer layers at runtime")
final_layer_index, final_layer_name, final_layer = max(layer_candidates, key=lambda item: item[0])
hidden_dimension = int(model.config.text_config.hidden_size)
print("Runtime final language layer:", final_layer_name)
print("Runtime hidden dimension:", hidden_dimension)
if final_layer_index != 35 or hidden_dimension != 2048:
    raise RuntimeError(
        f"Unexpected language model structure: final_layer={final_layer_index}, hidden={hidden_dimension}"
    )

special_ids = set(processor.tokenizer.all_special_ids)
total = len(samples) * len(SEVERITIES)
completed = sum(1 for sample in samples for severity in SEVERITIES if key(str(sample["question_id"]), severity) in rows and key(str(sample["question_id"]), severity) in vectors)

for sample_index, sample in enumerate(samples, 1):
    question_id = str(sample["question_id"])
    image_id = str(sample["image_id"])
    original = Image.open(IMAGE_DIR / f"{image_id}.jpg").convert("RGB")
    box = sample["critical_objects"][0]["bbox_xyxy"]
    trajectory_changed = False

    for severity in SEVERITIES:
        item_key = key(question_id, severity)
        if item_key in rows and item_key in vectors:
            continue
        condition_number = completed + 1
        print(
            f"[{condition_number}/{total}] trajectory={sample_index}/{len(samples)} "
            f"qid={question_id} lambda={severity:.2f}"
        )
        image = corrupt(original, box, severity)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["question"] + " Answer using only a short answer."},
            ],
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        captured = []

        def hook_fn(module, hook_inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden) and hidden.ndim == 3:
                captured.append(hidden[0, -1].detach().float().cpu())

        handle = final_layer.register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    return_dict_in_generate=True, output_scores=True,
                )
        finally:
            handle.remove()

        generated_ids = generated.sequences[:, inputs.input_ids.shape[1]:]
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        log_probabilities = []
        entropies = []
        for step, scores in enumerate(generated.scores):
            token_id = int(generated_ids[0, step])
            if token_id in special_ids:
                continue
            log_probs = torch.log_softmax(scores[0].float(), dim=-1)
            probabilities = log_probs.exp()
            log_probabilities.append(float(log_probs[token_id]))
            entropies.append(float(-(probabilities * log_probs).sum()))
        if not log_probabilities:
            raise RuntimeError(f"No non-special answer tokens for {question_id} severity={severity}")
        if not captured:
            raise RuntimeError(f"No final-layer hidden state for {question_id} severity={severity}")
        hidden = captured[0].numpy().astype(np.float32)
        if hidden.shape != (hidden_dimension,):
            raise RuntimeError(f"Unexpected hidden shape: {hidden.shape}")

        prediction = normalize(answer)
        target = normalize(sample["answer"])
        row = {
            "question_id": question_id,
            "image_id": image_id,
            "category": sample["category"],
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "severity": severity,
            "generated_answer": answer,
            "answer": answer,
            "correctness": prediction == target,
            "correct": prediction == target,
            "c_seq": float(np.mean(log_probabilities)),
            "entropy": float(np.mean(entropies)),
            "semantic_token_count": len(log_probabilities),
        }
        rows[item_key] = row
        vectors[item_key] = hidden
        completed += 1
        trajectory_changed = True
        save_results(rows)
        print(
            f"answer={answer!r} gt={sample['answer']!r} correct={row['correct']} "
            f"c_seq={row['c_seq']:.4f} entropy={row['entropy']:.4f}"
        )
        del inputs, generated, generated_ids, captured
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if trajectory_changed:
        save_hidden(vectors)
        interim = summarize(rows, samples)
        SUMMARY_PATH.write_text(json.dumps(interim, indent=2), encoding="utf-8")
        print(f"Saved trajectory {sample_index}: results={len(rows)} hidden={len(vectors)}")

summary = summarize(rows, samples)
if summary["conditions"] != total or len(vectors) != total:
    raise RuntimeError(
        f"Incomplete scaled run: results={summary['conditions']} hidden={len(vectors)} expected={total}"
    )
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

ordered_rows = sorted(rows.values(), key=lambda row: (row["question_id"], row["severity"]))
features = []
metadata = []
for row in ordered_rows:
    item_key = key(row["question_id"], row["severity"])
    features.append(np.concatenate([
        vectors[item_key], np.asarray([row["c_seq"], row["entropy"]], dtype=np.float32)
    ]))
    metadata.append({
        "question_id": row["question_id"], "image_id": row["image_id"],
        "category": row["category"], "severity": row["severity"],
        "correct": row["correct"], "answer": row["answer"],
        "ground_truth": row["ground_truth"],
    })
feature_matrix = np.stack(features).astype(np.float32)
if feature_matrix.shape != (total, 2050):
    raise RuntimeError(f"Unexpected feature matrix shape: {feature_matrix.shape}")
np.savez_compressed(FEATURES_PATH, X=feature_matrix)
META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\nP10/P11 COMPLETE")
print("Feature matrix:", feature_matrix.shape)
print("EMVR:", summary["emvr"])
print("Clean-correct EMVR:", summary["clean_correct_emvr"])
print("Trajectory violation rate:", summary["trajectory_violation_rate"])
print("P10 outputs:", P5_DIR)
print("P11 outputs:", P6_DIR)
