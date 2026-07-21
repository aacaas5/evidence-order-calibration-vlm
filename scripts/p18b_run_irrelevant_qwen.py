import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_accepted.json")
CONTROL_BOXES = Path("results/scaled/p18/control_boxes.json")
CRITICAL_RESULTS = Path("results/scaled/p5/results.json")
IMAGE_DIR = Path("data/gqa/scaled_images")
RESULT_PATH = Path("results/scaled/p18/irrelevant_results.json")
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def normalize(text):
    text = re.sub(r"[^\w\s-]", "", str(text).lower().strip())
    return " ".join(text.split())


def corrupt(image, box, severity):
    if severity == 0:
        return image.copy()
    x1, y1, x2, y2 = map(float, box)
    scale = math.sqrt(severity)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_width = (x2 - x1) * scale / 2
    half_height = (y2 - y1) * scale / 2
    output = image.copy()
    ImageDraw.Draw(output).rectangle(
        [center_x - half_width, center_y - half_height,
         center_x + half_width, center_y + half_height],
        fill=(127, 127, 127),
    )
    return output


def condition_key(question_id, severity):
    return f"{question_id}|{severity:.2f}"


def save(rows):
    ordered = sorted(rows.values(), key=lambda row: (row["question_id"], row["severity"]))
    RESULT_PATH.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


samples = {
    str(sample["question_id"]): sample
    for sample in json.loads(MANIFEST.read_text(encoding="utf-8"))
}
controls = {
    str(row["question_id"]): row
    for row in json.loads(CONTROL_BOXES.read_text(encoding="utf-8"))
    if row["control_valid"]
}
critical_clean = {
    str(row["question_id"]): row
    for row in json.loads(CRITICAL_RESULTS.read_text(encoding="utf-8"))
    if float(row["severity"]) == 0.0 and str(row["question_id"]) in controls
}
if set(controls) != set(critical_clean):
    raise RuntimeError("Valid controls and cached critical clean rows do not match")

rows = {}
if RESULT_PATH.exists():
    for row in json.loads(RESULT_PATH.read_text(encoding="utf-8")):
        rows[condition_key(str(row["question_id"]), float(row["severity"]))] = row

# The lambda=0 image and prompt are exactly identical, so preserve the cached deterministic output.
for question_id, clean in critical_clean.items():
    item_key = condition_key(question_id, 0.0)
    if item_key in rows:
        continue
    control = controls[question_id]
    row = dict(clean)
    row.update({
        "intervention": "matched_irrelevant",
        "control_bbox": control["control_bbox"],
        "control_overlap_score": control["control_overlap_score"],
        "source": "cached_identical_clean_condition_from_results/scaled/p5/results.json",
    })
    rows[item_key] = row
save(rows)

planned = len(controls) * len(SEVERITIES)
print("P18 MATCHED IRRELEVANT QWEN")
print("Valid trajectories:", len(controls))
print("Planned conditions:", planned)
print("Cached/completed conditions:", len(rows))

processor = AutoProcessor.from_pretrained(MODEL)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype="auto", device_map="auto"
)
model.eval()
special_ids = set(processor.tokenizer.all_special_ids)

completed = len(rows)
for trajectory_index, question_id in enumerate(controls, 1):
    sample = samples[question_id]
    control = controls[question_id]
    image_id = str(sample["image_id"])
    original = Image.open(IMAGE_DIR / f"{image_id}.jpg").convert("RGB")
    for severity in SEVERITIES[1:]:
        item_key = condition_key(question_id, severity)
        if item_key in rows:
            continue
        print(
            f"[{completed + 1}/{planned}] trajectory={trajectory_index}/{len(controls)} "
            f"qid={question_id} lambda={severity:.2f}"
        )
        image = corrupt(original, control["control_bbox"], severity)
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
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
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
            "intervention": "matched_irrelevant",
            "control_bbox": control["control_bbox"],
            "control_overlap_score": control["control_overlap_score"],
            "source": "frozen_qwen_inference",
        }
        rows[item_key] = row
        completed += 1
        save(rows)
        print(
            f"answer={answer!r} gt={sample['answer']!r} correct={row['correct']} "
            f"c_seq={row['c_seq']:.4f} entropy={row['entropy']:.4f}"
        )
        del inputs, generated, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if len(rows) != planned:
    raise RuntimeError(f"Incomplete P18 inference: {len(rows)} of {planned}")
save(rows)
print("P18 IRRELEVANT INFERENCE COMPLETE")
print("Trajectories:", len(controls))
print("Conditions:", len(rows))
print("Saved:", RESULT_PATH)
