import json, math
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MANIFEST = Path("data/gqa/manifests/gqa_evidence_pilot_audited.json")
IMG_DIR = Path("data/gqa/pilot_images")
OUT_DIR = Path("results/p5a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
s = samples[0]

image = Image.open(IMG_DIR / f"{s['image_id']}.jpg").convert("RGB")
obj = s["critical_objects"][0]
x1, y1, x2, y2 = map(int, obj["bbox_xyxy"])

severities = [0, .25, .5, .75, 1.0]

def corrupt(img, lam):
    out = img.copy()
    if lam == 0:
        return out

    # progressively cover the critical box from its centre outward
    w = x2 - x1
    h = y2 - y1
    scale = math.sqrt(lam)

    cx, cy = (x1+x2)/2, (y1+y2)/2
    hw, hh = w*scale/2, h*scale/2

    box = [cx-hw, cy-hh, cx+hw, cy+hh]
    ImageDraw.Draw(out).rectangle(box, fill=(127,127,127))
    return out

print("Question:", s["question"])
print("Ground truth:", s["answer"])
print("Critical object:", obj["name"])

paths = []

for lam in severities:
    img = corrupt(image, lam)
    p = OUT_DIR / f"{s['image_id']}_lambda_{lam:.2f}.jpg"
    img.save(p, quality=95)
    paths.append((lam, p))
    print(f"lambda={lam:.2f} -> {p}")

print("\nLoading Qwen...")

processor = AutoProcessor.from_pretrained(MODEL)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

results = []

for lam, path in paths:

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(path.resolve())},
            {"type": "text",
             "text": s["question"] + " Answer using only a short answer."}
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
        g = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    ids = g.sequences[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        ids,
        skip_special_tokens=True
    )[0].strip()

    logps, entropies = [], []

    for i, score in enumerate(g.scores):
        lp = torch.log_softmax(score[0].float(), dim=-1)
        p = lp.exp()

        token = ids[0, i].item()

        # ignore generation terminator in reliability statistics
        if token == processor.tokenizer.eos_token_id:
            continue

        logps.append(lp[token].item())
        entropies.append(-(p * lp).sum().item())

    c_seq = sum(logps) / len(logps)
    entropy = sum(entropies) / len(entropies)

    correct = answer.lower().strip(" .") == s["answer"].lower().strip(" .")

    row = {
        "severity": lam,
        "answer": answer,
        "correct": correct,
        "c_seq": c_seq,
        "entropy": entropy
    }

    results.append(row)

    print(
        f"lambda={lam:.2f} | "
        f"answer={answer!r} | "
        f"correct={correct} | "
        f"c_seq={c_seq:.4f} | "
        f"H={entropy:.4f}"
    )

out = OUT_DIR / "trajectory.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")

violations = sum(
    results[i+1]["c_seq"] > results[i]["c_seq"]
    for i in range(len(results)-1)
)

print("\nAdjacent confidence violations:", violations, "/ 4")
print("Saved:", out)
