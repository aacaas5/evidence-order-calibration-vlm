import json, math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

MANIFEST = Path("data/gqa/manifests/gqa_evidence_pilot_audited.json")
IMAGE_DIR = Path("data/gqa/pilot_images")
P5B_RESULTS = Path("results/p5b/results.json")

OUT_DIR = Path("results/p6a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = OUT_DIR / "hidden_features.npz"
META_PATH = OUT_DIR / "hidden_features_meta.json"

SEVERITIES = [0.0, 0.25, 0.50, 0.75, 1.0]


def corrupt(image, box, severity):
    if severity == 0:
        return image.copy()

    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    scale = math.sqrt(severity)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = w * scale / 2, h * scale / 2

    out = image.copy()

    ImageDraw.Draw(out).rectangle(
        [cx - hw, cy - hh, cx + hw, cy + hh],
        fill=(127, 127, 127)
    )

    return out


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
samples = {
    str(s["question_id"]): s
    for s in manifest
    if s.get("audit_status") == "accept"
}

p5b = json.loads(P5B_RESULTS.read_text(encoding="utf-8"))

print("Accepted samples:", len(samples))
print("Conditions:", len(p5b))

processor = AutoProcessor.from_pretrained(MODEL)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

layers = []
for name, module in model.named_modules():
    parts = name.split(".")
    if len(parts) >= 2 and parts[-2] == "layers":
        try:
            layers.append((int(parts[-1]), name, module))
        except ValueError:
            pass

layers.sort(key=lambda x: x[0])
_, layer_name, final_layer = layers[-1]

print("Final layer:", layer_name)

features = []
metadata = []

for i, row in enumerate(p5b, 1):
    qid = str(row["question_id"])
    severity = float(row["severity"])

    sample = samples[qid]
    obj = sample["critical_objects"][0]

    image = Image.open(
        IMAGE_DIR / f"{sample['image_id']}.jpg"
    ).convert("RGB")

    image = corrupt(
        image,
        obj["bbox_xyxy"],
        severity
    )

    captured = []

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(hidden) and hidden.ndim == 3:
            captured.append(
                hidden[0, -1, :].detach().float().cpu()
            )

    handle = final_layer.register_forward_hook(hook_fn)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": sample["question"] + " Answer using only a short answer."
            }
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False
        )

    handle.remove()

    if not captured:
        raise RuntimeError(f"No hidden state captured for {qid} {severity}")

    h = captured[0].numpy()

    feature = np.concatenate([
        h,
        np.array(
            [row["c_seq"], row["entropy"]],
            dtype=np.float32
        )
    ])

    features.append(feature)

    metadata.append({
        "question_id": qid,
        "image_id": row["image_id"],
        "severity": severity,
        "correct": bool(row["correct"]),
        "answer": row["answer"],
        "ground_truth": row["ground_truth"]
    })

    print(
        f"[{i}/{len(p5b)}] "
        f"qid={qid} "
        f"lambda={severity:.2f} "
        f"dim={feature.shape[0]}"
    )

    del inputs, generated

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


X = np.stack(features).astype(np.float32)

np.savez_compressed(
    FEATURES_PATH,
    X=X
)

META_PATH.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nP6A COMPLETE")
print("Feature matrix:", X.shape)
print("Saved:", FEATURES_PATH)
print("Saved:", META_PATH)
