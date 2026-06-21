import json
import math
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

MANIFEST = Path(
    "data/gqa/manifests/gqa_evidence_pilot_audited.json"
)

IMAGE_DIR = Path("data/gqa/pilot_images")

OUT_DIR = Path("results/p5b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUT_DIR / "results.json"
SUMMARY_PATH = OUT_DIR / "summary.json"

SEVERITIES = [0.0, 0.25, 0.50, 0.75, 1.0]


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return " ".join(text.split())


def corrupt(image, box, severity):
    if severity == 0:
        return image.copy()

    x1, y1, x2, y2 = map(int, box)

    w = x2 - x1
    h = y2 - y1

    scale = math.sqrt(severity)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    hw = w * scale / 2
    hh = h * scale / 2

    out = image.copy()

    ImageDraw.Draw(out).rectangle(
        [cx - hw, cy - hh, cx + hw, cy + hh],
        fill=(127, 127, 127)
    )

    return out


def load_manifest():
    data = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    # Support either a list or a wrapped manifest.
    if isinstance(data, dict):
        for key in ("samples", "results", "rows"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    accepted = []

    for sample in data:
        decision = str(
            sample.get(
                "audit_status",
                sample.get(
                    "audit",
                    sample.get(
                        "decision",
                        sample.get("audit_decision", "")
                    )
                )
            )
        ).lower()

        if decision == "accept":
            accepted.append(sample)

    return accepted


def load_existing():
    if not RESULT_PATH.exists():
        return []

    return json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )


def save(results):
    RESULT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )


print("=" * 72)
print("PROJECT 3 - P5B REAL GQA PILOT")
print("=" * 72)

samples = load_manifest()

print("Accepted samples:", len(samples))

if not samples:
    raise RuntimeError(
        "No accepted samples found in audited manifest."
    )

results = load_existing()

completed = {
    (
        str(r["question_id"]),
        float(r["severity"])
    )
    for r in results
}

print("Previously completed conditions:", len(completed))
print("Total planned conditions:", len(samples) * len(SEVERITIES))


print("\nLoading Qwen...")

processor = AutoProcessor.from_pretrained(MODEL)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

print("Model loaded.")

condition_number = len(completed)
total = len(samples) * len(SEVERITIES)


for sample in samples:

    qid = str(sample["question_id"])
    image_id = str(sample["image_id"])

    image_path = IMAGE_DIR / f"{image_id}.jpg"

    if not image_path.exists():
        print("MISSING IMAGE:", image_path)
        continue

    critical_objects = sample.get("critical_objects", [])

    if not critical_objects:
        print("NO CRITICAL OBJECT:", qid)
        continue

    obj = critical_objects[0]

    box = obj["bbox_xyxy"]

    original = Image.open(image_path).convert("RGB")

    for severity in SEVERITIES:

        key = (qid, float(severity))

        if key in completed:
            continue

        condition_number += 1

        print(
            f"\n[{condition_number}/{total}] "
            f"QID={qid} lambda={severity:.2f}"
        )

        image = corrupt(
            original,
            box,
            severity
        )

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text":
                        sample["question"]
                        + " Answer using only a short answer."
                }
            ]
        }]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():

            generated = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = generated.sequences[
            :,
            inputs.input_ids.shape[1]:
        ]

        answer = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        log_probs = []
        entropies = []

        for step, score in enumerate(generated.scores):

            lp = torch.log_softmax(
                score[0].float(),
                dim=-1
            )

            probs = lp.exp()

            token_id = generated_ids[0, step].item()

            if token_id == processor.tokenizer.eos_token_id:
                continue

            log_probs.append(
                lp[token_id].item()
            )

            entropies.append(
                -(probs * lp).sum().item()
            )

        if not log_probs:
            print("No semantic tokens generated.")
            continue

        c_seq = sum(log_probs) / len(log_probs)

        entropy = (
            sum(entropies) / len(entropies)
        )

        prediction = normalize(answer)
        target = normalize(sample["answer"])

        correct = prediction == target

        peak_vram = None

        if torch.cuda.is_available():
            peak_vram = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

        row = {
            "question_id": qid,
            "image_id": image_id,
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "critical_object": obj.get("name"),
            "severity": severity,
            "answer": answer,
            "correct": correct,
            "c_seq": c_seq,
            "entropy": entropy,
            "peak_vram_gb": peak_vram
        }

        results.append(row)
        completed.add(key)

        save(results)

        print(
            f"answer={answer!r} | "
            f"GT={sample['answer']!r} | "
            f"correct={correct} | "
            f"c_seq={c_seq:.4f} | "
            f"H={entropy:.4f}"
        )

        del inputs
        del generated
        del generated_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("\n" + "=" * 72)
print("AGGREGATE PILOT ANALYSIS")
print("=" * 72)


severity_summary = {}

for severity in SEVERITIES:

    rows = [
        r for r in results
        if float(r["severity"]) == severity
    ]

    if not rows:
        continue

    accuracy = (
        sum(r["correct"] for r in rows)
        / len(rows)
    )

    mean_confidence = (
        sum(r["c_seq"] for r in rows)
        / len(rows)
    )

    mean_entropy = (
        sum(r["entropy"] for r in rows)
        / len(rows)
    )

    severity_summary[str(severity)] = {
        "n": len(rows),
        "accuracy": accuracy,
        "mean_c_seq": mean_confidence,
        "mean_entropy": mean_entropy
    }

    print(
        f"lambda={severity:.2f} | "
        f"N={len(rows):2d} | "
        f"accuracy={accuracy:.3f} | "
        f"c_seq={mean_confidence:.4f} | "
        f"H={mean_entropy:.4f}"
    )


trajectories = {}

for row in results:
    trajectories.setdefault(
        row["question_id"],
        []
    ).append(row)


adjacent_violations = 0
adjacent_pairs = 0

trajectory_violations = 0
complete_trajectories = 0


for qid, rows in trajectories.items():

    rows = sorted(
        rows,
        key=lambda r: r["severity"]
    )

    if len(rows) != len(SEVERITIES):
        continue

    complete_trajectories += 1
    has_violation = False

    for left, right in zip(
        rows[:-1],
        rows[1:]
    ):

        adjacent_pairs += 1

        if right["c_seq"] > left["c_seq"]:
            adjacent_violations += 1
            has_violation = True

    if has_violation:
        trajectory_violations += 1


emvr = (
    adjacent_violations / adjacent_pairs
    if adjacent_pairs else None
)

trajectory_violation_rate = (
    trajectory_violations / complete_trajectories
    if complete_trajectories else None
)


summary = {
    "accepted_samples": len(samples),
    "conditions": len(results),
    "complete_trajectories": complete_trajectories,
    "severity_summary": severity_summary,
    "adjacent_pairs": adjacent_pairs,
    "adjacent_violations": adjacent_violations,
    "emvr": emvr,
    "trajectories_with_violation": trajectory_violations,
    "trajectory_violation_rate": trajectory_violation_rate
}


SUMMARY_PATH.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)


print("\nComplete trajectories:", complete_trajectories)

print(
    "Adjacent violations:",
    adjacent_violations,
    "/",
    adjacent_pairs
)

print(
    "Pilot EMVR:",
    round(emvr, 4)
    if emvr is not None else "N/A"
)

print(
    "Trajectories with >=1 violation:",
    trajectory_violations,
    "/",
    complete_trajectories
)

if trajectory_violation_rate is not None:
    print(
        "Trajectory violation rate:",
        round(trajectory_violation_rate, 4)
    )

print("\nSaved:", RESULT_PATH)
print("Saved:", SUMMARY_PATH)

print("\nP5B COMPLETE.")
