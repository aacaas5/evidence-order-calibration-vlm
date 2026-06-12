import os
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "QWEN_VL_MODEL",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

OUTPUT_DIR = Path("results/figures")
RAW_DIR = Path("results/raw")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

QUESTION = "What color is the square? Answer with one word."

GROUND_TRUTH = "red"

SEVERITIES = [0.0, 0.5, 1.0]

IMAGE_SIZE = 336

# Exact question-critical region:
# red square coordinates
CRITICAL_BOX = (55, 105, 155, 205)


def gb(x):
    return round(x / 1024**3, 2)


# ============================================================
# CREATE CONTROLLED BASE IMAGE
# ============================================================

print("=" * 74)
print("PROJECT 3 - P3 FIRST CONTROLLED EVIDENCE-LOSS PILOT")
print("=" * 74)

print("\n[1] Creating controlled image...")

base = Image.new(
    "RGB",
    (IMAGE_SIZE, IMAGE_SIZE),
    "white",
)

draw = ImageDraw.Draw(base)

# Question-critical RED square
draw.rectangle(
    CRITICAL_BOX,
    fill=(220, 30, 30),
)

# Irrelevant BLUE circle
draw.ellipse(
    (220, 110, 300, 190),
    fill=(30, 80, 220),
)

base_path = OUTPUT_DIR / "p3_clean.png"
base.save(base_path)

print("Clean image saved:", base_path)


# ============================================================
# TARGETED NESTED EVIDENCE REMOVAL
# ============================================================

def remove_evidence(image, bbox, severity):
    """
    Remove an increasing AREA fraction of the critical region.

    severity = 0.0 -> nothing removed
    severity = 0.5 -> ~50% critical-region area removed
    severity = 1.0 -> entire critical region removed

    Removal uses the white image background, so we do not
    introduce a new artificial colour/object.
    """

    output = image.copy()

    if severity <= 0:
        return output

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    # If we want lambda fraction of AREA removed,
    # each side of the centered rectangle scales by sqrt(lambda).
    scale = math.sqrt(severity)

    cut_w = width * scale
    cut_h = height * scale

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    cut_box = (
        int(cx - cut_w / 2),
        int(cy - cut_h / 2),
        int(cx + cut_w / 2),
        int(cy + cut_h / 2),
    )

    draw = ImageDraw.Draw(output)

    draw.rectangle(
        cut_box,
        fill="white",
    )

    return output


trajectory_paths = []

for severity in SEVERITIES:

    corrupted = remove_evidence(
        base,
        CRITICAL_BOX,
        severity,
    )

    filename = (
        OUTPUT_DIR /
        f"p3_critical_lambda_{severity:.2f}.png"
    )

    corrupted.save(filename)

    trajectory_paths.append(
        (severity, filename)
    )

    print(
        f"lambda={severity:.2f} -> {filename}"
    )


# ============================================================
# LOAD PROCESSOR + FROZEN VLM
# ============================================================

print("\n[2] Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("[3] Loading frozen Qwen...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

print("Model loaded.")

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Allocated after load:",
        gb(torch.cuda.memory_allocated()),
        "GB",
    )


# ============================================================
# LOCATE FINAL LANGUAGE LAYER
# ============================================================

candidate_layers = []

for name, module in model.named_modules():

    parts = name.split(".")

    if (
        len(parts) >= 2
        and parts[-2] == "layers"
    ):
        try:
            idx = int(parts[-1])

            candidate_layers.append(
                (idx, name, module)
            )

        except ValueError:
            pass


if not candidate_layers:
    raise RuntimeError(
        "Could not locate Qwen language layers."
    )


candidate_layers.sort(
    key=lambda x: x[0]
)

final_layer_index, final_layer_name, final_layer = (
    candidate_layers[-1]
)

print(
    "Using final language layer:",
    final_layer_name
)


# ============================================================
# SINGLE-SAMPLE INFERENCE FUNCTION
# ============================================================

def evaluate_image(
    image_path,
    severity,
):

    captured_states = []

    def hook_fn(module, inputs, output):

        hidden = (
            output[0]
            if isinstance(output, tuple)
            else output
        )

        if (
            torch.is_tensor(hidden)
            and hidden.ndim == 3
        ):
            state = (
                hidden[0, -1, :]
                .detach()
                .float()
                .cpu()
            )

            captured_states.append(
                state
            )


    handle = final_layer.register_forward_hook(
        hook_fn
    )


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(
                        image_path.resolve()
                    ),
                },
                {
                    "type": "text",
                    "text": QUESTION,
                },
            ],
        }
    ]


    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    image_inputs, video_inputs = (
        process_vision_info(messages)
    )


    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


    inputs = inputs.to(model.device)

    prompt_length = (
        inputs.input_ids.shape[1]
    )


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )


    handle.remove()


    generated_ids = (
        outputs.sequences[0][
            prompt_length:
        ]
    )


    answer = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


    special_ids = set(
        processor.tokenizer.all_special_ids
    )


    rows = []

    for step, score_tensor in enumerate(
        outputs.scores
    ):

        if step >= len(generated_ids):
            break

        token_id = (
            generated_ids[step].item()
        )

        if token_id in special_ids:
            continue


        logits = (
            score_tensor[0]
            .float()
        )


        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )

        probs = torch.exp(
            log_probs
        )


        token_log_prob = (
            log_probs[token_id]
            .item()
        )

        token_probability = (
            probs[token_id]
            .item()
        )

        entropy = -(
            probs * log_probs
        ).sum().item()


        rows.append(
            {
                "token_probability":
                    token_probability,

                "token_log_probability":
                    token_log_prob,

                "entropy":
                    entropy,
            }
        )


    if not rows:
        raise RuntimeError(
            "No semantic tokens generated."
        )


    c_seq = np.mean(
        [
            x["token_log_probability"]
            for x in rows
        ]
    )


    avg_entropy = np.mean(
        [
            x["entropy"]
            for x in rows
        ]
    )


    geom_prob = math.exp(
        c_seq
    )


    if not captured_states:
        raise RuntimeError(
            "Hidden state not captured."
        )


    first_hidden = (
        captured_states[0]
    )


    correct = (
        answer.lower()
        .strip(" .")
        == GROUND_TRUTH
    )


    result = {
        "severity": severity,
        "image": str(image_path),
        "question": QUESTION,
        "ground_truth": GROUND_TRUTH,
        "answer": answer,
        "correct": bool(correct),
        "c_seq": float(c_seq),
        "geometric_mean_probability":
            float(geom_prob),
        "average_entropy":
            float(avg_entropy),
        "hidden_dimension":
            int(first_hidden.numel()),
        "hidden_norm":
            float(
                torch.linalg.vector_norm(
                    first_hidden
                ).item()
            ),
        "peak_vram_gb":
            gb(
                torch.cuda.max_memory_allocated()
            )
            if torch.cuda.is_available()
            else None,
    }


    return result, first_hidden


# ============================================================
# RUN EVIDENCE TRAJECTORY
# ============================================================

print("\n[4] Running evidence trajectory...")

results = []

hidden_vectors = {}

for severity, image_path in trajectory_paths:

    print(
        f"\n--- lambda = {severity:.2f} ---"
    )

    result, hidden = evaluate_image(
        image_path,
        severity,
    )

    hidden_vectors[severity] = hidden

    results.append(
        result
    )

    print(
        "Answer:",
        result["answer"]
    )

    print(
        "Correct:",
        result["correct"]
    )

    print(
        "c_seq:",
        round(
            result["c_seq"],
            6,
        )
    )

    print(
        "Geom. probability:",
        round(
            result[
                "geometric_mean_probability"
            ],
            6,
        )
    )

    print(
        "Entropy:",
        round(
            result[
                "average_entropy"
            ],
            6,
        )
    )

    print(
        "Peak VRAM:",
        result["peak_vram_gb"],
        "GB"
    )


# ============================================================
# HIDDEN-STATE SIMILARITY TO CLEAN STATE
# ============================================================

print("\n" + "=" * 74)
print("HIDDEN-STATE CHANGE")
print("=" * 74)

clean_hidden = hidden_vectors[0.0]

for severity in SEVERITIES:

    h = hidden_vectors[severity]

    similarity = (
        torch.nn.functional.cosine_similarity(
            clean_hidden.unsqueeze(0),
            h.unsqueeze(0),
        ).item()
    )

    for result in results:

        if result["severity"] == severity:

            result[
                "hidden_cosine_to_clean"
            ] = similarity


    print(
        f"lambda={severity:.2f} | "
        f"cosine similarity to clean = "
        f"{similarity:.6f}"
    )


# ============================================================
# SIMPLE MONOTONICITY CHECK
# ============================================================

print("\n" + "=" * 74)
print("NATIVE CONFIDENCE ORDER CHECK")
print("=" * 74)

violations = 0

for i in range(
    len(results) - 1
):

    current = results[i]
    next_result = results[i + 1]

    # c_seq closer to zero means MORE confident.
    #
    # Therefore if next c_seq > current c_seq,
    # confidence increased despite more evidence loss.

    violation = (
        next_result["c_seq"]
        >
        current["c_seq"]
    )

    if violation:
        violations += 1

    print(
        f"{current['severity']:.2f}"
        f" -> "
        f"{next_result['severity']:.2f}"
        f" | "
        f"{'VIOLATION' if violation else 'OK'}"
    )


possible_pairs = (
    len(results) - 1
)

pilot_emvr = (
    violations /
    possible_pairs
)

print(
    "\nPilot EMVR:",
    round(pilot_emvr, 4)
)


# ============================================================
# SAVE RESULTS
# ============================================================

json_path = (
    RAW_DIR /
    "p3_first_evidence_trajectory.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


print("\n" + "=" * 74)
print("SUMMARY")
print("=" * 74)

print(
    f"{'lambda':<10}"
    f"{'answer':<20}"
    f"{'correct':<10}"
    f"{'c_seq':<12}"
    f"{'entropy':<12}"
)

for result in results:

    print(
        f"{result['severity']:<10.2f}"
        f"{result['answer']:<20}"
        f"{str(result['correct']):<10}"
        f"{result['c_seq']:<12.4f}"
        f"{result['average_entropy']:<12.4f}"
    )


print(
    "\nSaved result:",
    json_path
)

print("\nP3 PILOT COMPLETE.")
