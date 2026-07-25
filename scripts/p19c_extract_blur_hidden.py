import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

IMAGE_DIR = Path("data/gqa/scaled_images")
MANIFEST = Path(
    "data/gqa/manifests/gqa_evidence_scaled_accepted.json"
)
RESULTS = Path(
    "results/scaled/p19/blur_results.json"
)

OUT = Path("results/scaled/p19")
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = OUT / "blur_hidden_features.npz"
META_PATH = OUT / "blur_hidden_meta.json"

CHECKPOINT_X = OUT / "blur_hidden_partial.npy"
CHECKPOINT_META = OUT / "blur_hidden_partial_meta.json"


# ============================================================
# Blur
# ============================================================

def blur_radius(box, severity):
    x1, y1, x2, y2 = map(float, box)

    base = min(
        x2 - x1,
        y2 - y1
    )

    return min(
        severity * 0.15 * base,
        24.0
    )


def apply_blur(image, box, severity):

    if severity == 0:
        return image.copy()

    x1, y1, x2, y2 = map(
        int,
        box
    )

    radius = blur_radius(
        box,
        severity
    )

    out = image.copy()

    crop = out.crop(
        (x1, y1, x2, y2)
    )

    crop = crop.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )

    out.paste(
        crop,
        (x1, y1)
    )

    return out


# ============================================================
# Load metadata
# ============================================================

samples = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

rows = json.loads(
    RESULTS.read_text(
        encoding="utf-8"
    )
)

sample_map = {
    str(s["question_id"]): s
    for s in samples
}

rows = sorted(
    rows,
    key=lambda r: (
        str(r["question_id"]),
        float(r["severity"])
    )
)

print("Conditions:", len(rows))


# ============================================================
# Resume
# ============================================================

if (
    CHECKPOINT_X.exists()
    and CHECKPOINT_META.exists()
):

    features = list(
        np.load(CHECKPOINT_X)
    )

    meta = json.loads(
        CHECKPOINT_META.read_text(
            encoding="utf-8"
        )
    )

else:

    features = []
    meta = []


start = len(features)

print(
    "Previously completed:",
    start
)


# ============================================================
# Model
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
)

model.eval()


# ============================================================
# Robustly locate language transformer layers
# ============================================================

def get_language_layers(model):

    candidates = [
        (
            "model.model.language_model.layers",
            lambda m:
                m.model.language_model.layers
        ),
        (
            "model.language_model.layers",
            lambda m:
                m.language_model.layers
        ),
        (
            "model.model.layers",
            lambda m:
                m.model.layers
        ),
    ]

    for name, getter in candidates:

        try:
            layers = getter(model)

            if len(layers) > 0:
                return name, layers

        except (AttributeError, TypeError):
            pass

    raise RuntimeError(
        "Could not locate Qwen language "
        "transformer layers."
    )


layer_path, layers = get_language_layers(
    model
)

final_layer = layers[-1]

print(
    "Language layer path:",
    layer_path
)

print(
    "Final language layer index:",
    len(layers) - 1
)


# ============================================================
# Hook
# ============================================================

captured = {}


def hook_fn(module, inputs, output):

    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output

    captured["hidden"] = (
        hidden.detach()
        .float()
        .cpu()
    )


handle = final_layer.register_forward_hook(
    hook_fn
)


# ============================================================
# Extract prompt decision state
# ============================================================

for i in range(
    start,
    len(rows)
):

    r = rows[i]

    qid = str(
        r["question_id"]
    )

    s = sample_map[qid]

    severity = float(
        r["severity"]
    )

    image = Image.open(
        IMAGE_DIR
        / f"{s['image_id']}.jpg"
    ).convert("RGB")

    image = apply_blur(
        image,
        s["critical_objects"][0][
            "bbox_xyxy"
        ],
        severity
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # user prompt ONLY.
    #
    # The final hidden state of this prompt is the
    # representation used to predict the FIRST answer token.
    # --------------------------------------------------------

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

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # With device_map="auto", move inputs to
    # the embedding/input device.
    input_device = next(
        model.parameters()
    ).device

    inputs = {
        k: (
            v.to(input_device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in inputs.items()
    }


    captured.clear()

    with torch.inference_mode():

        model(
            **inputs,
            use_cache=False,
            return_dict=True
        )


    if "hidden" not in captured:

        raise RuntimeError(
            f"No hidden state captured "
            f"for qid={qid}"
        )


    hidden = captured["hidden"]

    # Last prompt position:
    # state immediately before first answer token.
    vec = hidden[
        0,
        -1,
        :
    ].numpy()


    if vec.shape[0] != 2048:

        raise RuntimeError(
            "Unexpected hidden dimension: "
            f"{vec.shape}"
        )


    feature = np.concatenate(
        [
            vec.astype(
                np.float32
            ),
            np.array(
                [
                    float(r["c_seq"]),
                    float(r["entropy"])
                ],
                dtype=np.float32
            )
        ]
    )


    if feature.shape[0] != 2050:

        raise RuntimeError(
            "Unexpected feature dimension: "
            f"{feature.shape}"
        )


    features.append(feature)

    meta.append({
        "question_id": qid,
        "image_id": str(
            r["image_id"]
        ),
        "severity": severity,
        "category": r.get(
            "category"
        ),
        "correct": bool(
            r["correct"]
        ),
        "answer": r["answer"],
        "ground_truth":
            r["ground_truth"],
        "c_seq": float(
            r["c_seq"]
        ),
        "entropy": float(
            r["entropy"]
        )
    })


    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    np.save(
        CHECKPOINT_X,
        np.asarray(
            features,
            dtype=np.float32
        )
    )

    CHECKPOINT_META.write_text(
        json.dumps(
            meta,
            indent=2
        ),
        encoding="utf-8"
    )


    print(
        f"[{i+1}/{len(rows)}] "
        f"qid={qid} "
        f"lambda={severity:.2f} "
        f"dim={feature.shape[0]}"
    )


    del inputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


handle.remove()


# ============================================================
# Final save
# ============================================================

X = np.asarray(
    features,
    dtype=np.float32
)

np.savez_compressed(
    FEATURE_PATH,
    X=X
)

META_PATH.write_text(
    json.dumps(
        meta,
        indent=2
    ),
    encoding="utf-8"
)


print("\n" + "=" * 72)
print("P19C BLUR HIDDEN FEATURES")
print("=" * 72)

print(
    "Conditions:",
    len(meta)
)

print(
    "Feature matrix:",
    X.shape
)

print(
    "Expected:",
    (len(rows), 2050)
)

print(
    "Saved:",
    FEATURE_PATH
)

print(
    "Saved:",
    META_PATH
)

print("\nP19C COMPLETE")
