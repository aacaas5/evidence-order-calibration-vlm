import json, math, re
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_accepted.json")
IMAGE_DIR = Path("data/gqa/scaled_images")

CLEAN_CACHE = Path("results/scaled/p5/results.json")

OUT = Path("results/scaled/p19")
OUT.mkdir(parents=True, exist_ok=True)

RESULT_PATH = OUT / "blur_results.json"
SUMMARY_PATH = OUT / "blur_summary.json"

SEVERITIES = [0.0, 0.25, 0.50, 0.75, 1.0]


def normalize(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())


def radius_for(box, lam):
    x1, y1, x2, y2 = map(float, box)
    base = min(x2 - x1, y2 - y1)
    return min(lam * 0.15 * base, 24.0)


def local_blur(image, box, lam):
    if lam == 0:
        return image.copy(), 0.0

    x1, y1, x2, y2 = map(int, box)
    r = radius_for(box, lam)

    out = image.copy()
    crop = out.crop((x1, y1, x2, y2))
    crop = crop.filter(ImageFilter.GaussianBlur(radius=r))
    out.paste(crop, (x1, y1))

    return out, r


samples = json.loads(
    MANIFEST.read_text(encoding="utf-8")
)

sample_by_qid = {
    str(s["question_id"]): s
    for s in samples
}


# ------------------------------------------------------------
# Clean cache from masking experiment
# ------------------------------------------------------------

clean_cache = {}

if CLEAN_CACHE.exists():
    old = json.loads(
        CLEAN_CACHE.read_text(encoding="utf-8")
    )

    for r in old:
        if float(r["severity"]) == 0.0:
            clean_cache[str(r["question_id"])] = r


# ------------------------------------------------------------
# Resume support
# ------------------------------------------------------------

if RESULT_PATH.exists():
    results = json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )
else:
    results = []

done = {
    (
        str(r["question_id"]),
        float(r["severity"])
    )
    for r in results
}

print("Trajectories:", len(samples))
print("Planned conditions:", len(samples) * 5)
print("Previously completed:", len(done))


# ------------------------------------------------------------
# Load frozen Qwen
# ------------------------------------------------------------

processor = AutoProcessor.from_pretrained(MODEL)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

special_ids = set(
    processor.tokenizer.all_special_ids
)

print("Model loaded.")


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

counter = len(done)
total = len(samples) * 5

for s in samples:

    qid = str(s["question_id"])
    box = s["critical_objects"][0]["bbox_xyxy"]

    for lam in SEVERITIES:

        key = (qid, float(lam))

        if key in done:
            continue

        counter += 1

        # ----------------------------------------------------
        # Clean condition: reuse cached identical result
        # ----------------------------------------------------

        if lam == 0.0 and qid in clean_cache:

            old = clean_cache[qid]

            row = {
                "question_id": qid,
                "image_id": str(s["image_id"]),
                "category": s.get("category"),
                "question": s["question"],
                "ground_truth": s["answer"],
                "severity": 0.0,
                "blur_radius": 0.0,
                "answer": old["answer"],
                "correct": bool(old["correct"]),
                "c_seq": float(old["c_seq"]),
                "entropy": float(old["entropy"]),
                "source": "cached_clean_mask_result"
            }

            results.append(row)

            RESULT_PATH.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8"
            )

            print(
                f"[{counter}/{total}] "
                f"qid={qid} lambda=0.00 cached"
            )

            continue


        # ----------------------------------------------------
        # Build blurred image
        # ----------------------------------------------------

        image = Image.open(
            IMAGE_DIR / f"{s['image_id']}.jpg"
        ).convert("RGB")

        image, blur_radius = local_blur(
            image,
            box,
            lam
        )


        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

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
                        s["question"]
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


        # ----------------------------------------------------
        # Generate with token scores
        # ----------------------------------------------------

        with torch.inference_mode():

            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )


        prompt_len = inputs["input_ids"].shape[1]

        generated_ids = out.sequences[
            :,
            prompt_len:
        ][0]


        # ----------------------------------------------------
        # Decode answer
        # ----------------------------------------------------

        answer = processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()


        # ----------------------------------------------------
        # Confidence + entropy
        # Exclude ALL special tokens
        # ----------------------------------------------------

        logps = []
        entropies = []

        for token_id, score in zip(
            generated_ids.tolist(),
            out.scores
        ):

            if token_id in special_ids:
                continue

            logits = score[0].float()

            probs = torch.softmax(
                logits,
                dim=-1
            )

            p = probs[token_id].clamp_min(1e-12)

            logps.append(
                torch.log(p).item()
            )

            entropy = -(
                probs
                * torch.log(
                    probs.clamp_min(1e-12)
                )
            ).sum().item()

            entropies.append(entropy)


        if not logps:
            raise RuntimeError(
                f"No semantic tokens for {qid}, lambda={lam}"
            )


        c_seq = sum(logps) / len(logps)
        entropy = sum(entropies) / len(entropies)

        correct = (
            normalize(answer)
            == normalize(s["answer"])
        )


        row = {
            "question_id": qid,
            "image_id": str(s["image_id"]),
            "category": s.get("category"),
            "question": s["question"],
            "ground_truth": s["answer"],
            "severity": float(lam),
            "blur_radius": float(blur_radius),
            "answer": answer,
            "correct": bool(correct),
            "c_seq": float(c_seq),
            "entropy": float(entropy),
            "semantic_token_count": len(logps),
            "source": "frozen_qwen_blur"
        }

        results.append(row)

        RESULT_PATH.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8"
        )


        print(
            f"[{counter}/{total}] "
            f"qid={qid} "
            f"lambda={lam:.2f} "
            f"ans='{answer}' "
            f"ok={int(correct)} "
            f"c={c_seq:.4f} "
            f"H={entropy:.4f}"
        )


        del inputs, out

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------
# Aggregate blur phenomenon
# ------------------------------------------------------------

summary = {}

for lam in SEVERITIES:

    subset = [
        r for r in results
        if float(r["severity"]) == lam
    ]

    summary[str(lam)] = {
        "n": len(subset),
        "accuracy": sum(
            r["correct"] for r in subset
        ) / len(subset),
        "mean_c_seq": sum(
            r["c_seq"] for r in subset
        ) / len(subset),
        "mean_entropy": sum(
            r["entropy"] for r in subset
        ) / len(subset),
    }


# ------------------------------------------------------------
# EMVR
# ------------------------------------------------------------

violations = 0
pairs = 0
bad_trajectories = 0

clean_correct_v = 0
clean_correct_pairs = 0
clean_correct_n = 0

for qid in sample_by_qid:

    traj = sorted(
        [
            r for r in results
            if str(r["question_id"]) == qid
        ],
        key=lambda x: float(x["severity"])
    )

    if len(traj) != 5:
        continue

    local_v = 0

    for a, b in zip(
        traj[:-1],
        traj[1:]
    ):

        if b["c_seq"] > a["c_seq"]:
            violations += 1
            local_v += 1

        pairs += 1

    if local_v:
        bad_trajectories += 1

    if traj[0]["correct"]:

        clean_correct_n += 1

        for a, b in zip(
            traj[:-1],
            traj[1:]
        ):

            if b["c_seq"] > a["c_seq"]:
                clean_correct_v += 1

            clean_correct_pairs += 1


report = {
    "trajectories": len(samples),
    "conditions": len(results),
    "severity_summary": summary,
    "emvr": violations / pairs,
    "trajectory_violation_rate":
        bad_trajectories / len(samples),
    "clean_correct_trajectories":
        clean_correct_n,
    "clean_correct_emvr":
        clean_correct_v / clean_correct_pairs
}


SUMMARY_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


print("\n" + "=" * 72)
print("P19B BLUR PHENOMENON")
print("=" * 72)

for lam in SEVERITIES:

    s = summary[str(lam)]

    print(
        f"lambda={lam:.2f} "
        f"accuracy={s['accuracy']:.4f} "
        f"c_seq={s['mean_c_seq']:.4f} "
        f"H={s['mean_entropy']:.4f}"
    )

print("\nEMVR:", round(report["emvr"], 4))

print(
    "Trajectory violation rate:",
    round(
        report["trajectory_violation_rate"],
        4
    )
)

print(
    "Clean-correct trajectories:",
    report["clean_correct_trajectories"]
)

print(
    "Clean-correct EMVR:",
    round(
        report["clean_correct_emvr"],
        4
    )
)

print("\nSaved:", RESULT_PATH)
print("Saved:", SUMMARY_PATH)

print("\nP19B COMPLETE")
