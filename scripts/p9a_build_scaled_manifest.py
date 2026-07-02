import json, re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("data/gqa")
OUT = ROOT / "manifests"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = 250

def find_file(name):
    hits = list(ROOT.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

QUESTIONS_PATH = find_file("val_balanced_questions.json")
SCENES_PATH = find_file("val_sceneGraphs.json")

print("Questions:", QUESTIONS_PATH)
print("Scenes:", SCENES_PATH)

questions = json.loads(
    QUESTIONS_PATH.read_text(encoding="utf-8")
)

scenes = json.loads(
    SCENES_PATH.read_text(encoding="utf-8")
)

id_re = re.compile(r"\((\d+)\)")

bad_terms = (
    " left of ", " right of ", " behind ", " in front of ",
    " above ", " below ", " next to ", " beside ", " between ",
    " near ", " farther ", " closer ", " same ", " larger ",
    " smaller ", " bigger ", " taller ", " shorter ",
    " how many ", " are there ", " any ", " both ",
)

def category(question):
    q = " " + question.lower().strip() + " "

    if "what color" in q or "what colour" in q:
        return "color"

    if (
        "what material" in q
        or "made of" in q
        or "made from" in q
    ):
        return "material"

    if "what shape" in q:
        return "shape"

    if (
        "what is this" in q
        or "what is the" in q
        or "what kind of" in q
        or "what type of" in q
        or "called" in q
    ):
        return "identity"

    return None

def object_ids(q):
    ids = []

    for step in q.get("semantic", []):
        arg = str(step.get("argument", ""))
        ids.extend(id_re.findall(arg))

    return sorted(set(ids))

def get_box(scene, oid):
    obj = scene.get("objects", {}).get(str(oid))

    if not obj:
        return None

    try:
        x = float(obj["x"])
        y = float(obj["y"])
        w = float(obj["w"])
        h = float(obj["h"])
    except Exception:
        return None

    if w <= 0 or h <= 0:
        return None

    return [x, y, x + w, y + h], obj

stats = Counter()
pool = defaultdict(list)

for qid, q in questions.items():

    text = str(q.get("question", "")).strip()
    cat = category(text)

    if cat is None:
        stats["unsupported_type"] += 1
        continue

    low = " " + text.lower() + " "

    if any(term in low for term in bad_terms):
        stats["relational_or_global"] += 1
        continue

    ids = object_ids(q)

    if len(ids) != 1:
        stats["not_single_object"] += 1
        continue

    image_id = str(q.get("imageId", ""))

    scene = scenes.get(image_id)

    if not scene:
        stats["missing_scene"] += 1
        continue

    width = float(scene.get("width", 0))
    height = float(scene.get("height", 0))

    if width <= 0 or height <= 0:
        stats["bad_image_size"] += 1
        continue

    box_info = get_box(scene, ids[0])

    if box_info is None:
        stats["missing_box"] += 1
        continue

    box, obj = box_info
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    area_ratio = (bw * bh) / (width * height)

    if not (0.015 <= area_ratio <= 0.35):
        stats["bad_area"] += 1
        continue

    if bw / width > 0.75 or bh / height > 0.75:
        stats["too_broad"] += 1
        continue

    aspect = max(bw / bh, bh / bw)

    if aspect > 6:
        stats["extreme_aspect"] += 1
        continue

    sample = {
        "question_id": str(qid),
        "image_id": image_id,
        "question": text,
        "answer": q.get("answer"),
        "category": cat,
        "critical_objects": [{
            "object_id": str(ids[0]),
            "name": obj.get("name"),
            "bbox_xyxy": [
                round(v, 2) for v in box
            ],
            "area_ratio": round(area_ratio, 6),
        }],
        "image_width": int(width),
        "image_height": int(height),
        "audit_status": "pending",
    }

    pool[cat].append(sample)
    stats["eligible"] += 1


# ------------------------------------------------------------
# Balanced selection with unique images
# ------------------------------------------------------------

categories = ["color", "material", "shape", "identity"]

selected = []
used_images = set()

while len(selected) < TARGET:

    added = False

    for cat in categories:

        while pool[cat]:

            s = pool[cat].pop(0)

            if s["image_id"] in used_images:
                continue

            selected.append(s)
            used_images.add(s["image_id"])
            added = True
            break

        if len(selected) >= TARGET:
            break

    if not added:
        break


manifest_path = OUT / "gqa_evidence_scaled_250.json"
stats_path = OUT / "gqa_evidence_scaled_250_stats.json"

manifest_path.write_text(
    json.dumps(selected, indent=2),
    encoding="utf-8"
)

selected_counts = Counter(
    x["category"] for x in selected
)

report = {
    "target": TARGET,
    "selected": len(selected),
    "unique_images": len(used_images),
    "selected_by_category": dict(selected_counts),
    "filter_stats": dict(stats),
}

stats_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 72)
print("P9A SCALED EVIDENCE MANIFEST")
print("=" * 72)

print("Target:", TARGET)
print("Selected:", len(selected))
print("Unique images:", len(used_images))

print("\nBy category:")
for k, v in selected_counts.items():
    print(f"{k}: {v}")

print("\nFilter statistics:")
for k, v in stats.most_common():
    print(f"{k}: {v}")

print("\nSaved:", manifest_path)
print("Saved:", stats_path)
print("\nP9A COMPLETE")
