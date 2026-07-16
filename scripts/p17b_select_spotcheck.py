import json, random
from pathlib import Path

random.seed(42)

MANIFEST = Path(
    "data/gqa/manifests/gqa_evidence_scaled_250_codex_audited.json"
)

OUT = Path("results/scaled/p17")
OUT.mkdir(parents=True, exist_ok=True)

data = json.loads(
    MANIFEST.read_text(encoding="utf-8")
)

sampled = random.sample(data, 10)

for i, s in enumerate(sampled, 1):
    obj = s["critical_objects"][0]

    print("=" * 70)
    print(f"[{i}/10]")
    print("Question ID:", s["question_id"])
    print("Image ID:", s["image_id"])
    print("Question:", s["question"])
    print("Answer:", s["answer"])
    print("Object:", obj.get("name"))
    print("BBox:", obj.get("bbox_xyxy"))
    print("Codex decision:", s.get("audit_status"))
    print("Reason:", s.get("audit_reason"))
    print(
        "Image:",
        f"data/gqa/scaled_images/{s['image_id']}.jpg"
    )

(OUT / "human_spotcheck_10.json").write_text(
    json.dumps(sampled, indent=2),
    encoding="utf-8"
)

print("\nSaved:", OUT / "human_spotcheck_10.json")
