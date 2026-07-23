import json
import math
import random
from pathlib import Path

from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt

MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_accepted.json")
IMAGE_DIR = Path("data/gqa/scaled_images")

OUT = Path("results/scaled/p19")
OUT.mkdir(parents=True, exist_ok=True)

SEVERITIES = [0.0, 0.25, 0.50, 0.75, 1.0]
SEED = 42

random.seed(SEED)


def blur_radius(box, severity):
    x1, y1, x2, y2 = map(float, box)

    w = x2 - x1
    h = y2 - y1

    base_dim = min(w, h)

    radius = severity * 0.15 * base_dim

    return min(radius, 24.0)


def apply_local_blur(image, box, severity):
    if severity == 0:
        return image.copy(), 0.0

    x1, y1, x2, y2 = map(int, box)

    radius = blur_radius(box, severity)

    out = image.copy()

    crop = out.crop((x1, y1, x2, y2))

    blurred = crop.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )

    out.paste(
        blurred,
        (x1, y1)
    )

    return out, radius


samples = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

print("Accepted trajectories:", len(samples))


# ------------------------------------------------------------
# Metadata verification
# ------------------------------------------------------------

records = []

for s in samples:

    box = s["critical_objects"][0]["bbox_xyxy"]

    row = {
        "question_id": str(s["question_id"]),
        "image_id": str(s["image_id"]),
        "question": s["question"],
        "ground_truth": s["answer"],
        "category": s.get("category"),
        "critical_bbox": box,
        "blur_radii": {
            str(lam): round(
                blur_radius(
                    box,
                    lam
                ),
                4
            )
            for lam in SEVERITIES
        }
    }

    records.append(row)


(OUT / "blur_manifest.json").write_text(
    json.dumps(
        records,
        indent=2
    ),
    encoding="utf-8"
)


# ------------------------------------------------------------
# QC examples
# ------------------------------------------------------------

qc_n = min(8, len(samples))

qc_samples = random.sample(
    samples,
    qc_n
)

fig, axes = plt.subplots(
    qc_n,
    5,
    figsize=(17, 3.2 * qc_n)
)

if qc_n == 1:
    axes = [axes]


for row_i, s in enumerate(qc_samples):

    image_path = (
        IMAGE_DIR
        / f"{s['image_id']}.jpg"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    box = s[
        "critical_objects"
    ][0]["bbox_xyxy"]

    question = s["question"]

    for col_i, lam in enumerate(
        SEVERITIES
    ):

        transformed, radius = apply_local_blur(
            image,
            box,
            lam
        )

        ax = axes[row_i][col_i]

        ax.imshow(transformed)
        ax.axis("off")

        ax.set_title(
            f"λ={lam:.2f}\nr={radius:.2f}px",
            fontsize=9
        )

        if col_i == 0:
            ax.set_ylabel(
                question[:55],
                fontsize=8
            )


plt.tight_layout()

QC_PATH = OUT / "blur_qc_examples.png"

plt.savefig(
    QC_PATH,
    dpi=180
)

plt.close()


print("\n" + "=" * 72)
print("P19A LOCAL BLUR TRAJECTORY SETUP")
print("=" * 72)

print("Trajectories:", len(samples))
print("Severities:", SEVERITIES)
print("Expected conditions:", len(samples) * len(SEVERITIES))

all_radii = []

for r in records:
    for lam, radius in r["blur_radii"].items():
        if float(lam) > 0:
            all_radii.append(radius)

print(
    "Mean nonzero blur radius:",
    round(
        sum(all_radii) / len(all_radii),
        4
    )
)

print(
    "Max blur radius:",
    round(
        max(all_radii),
        4
    )
)

print("Saved:", OUT / "blur_manifest.json")
print("Saved:", QC_PATH)

print("\nP19A COMPLETE")
