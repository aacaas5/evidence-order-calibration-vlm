import json, re
from pathlib import Path
from collections import Counter, defaultdict

RESULTS = Path("results/scaled/p5/results.json")
OUTDIR = Path("results/scaled/p17")
OUTDIR.mkdir(parents=True, exist_ok=True)

rows = json.loads(RESULTS.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# Official-style strict answer comparison
# ------------------------------------------------------------

def normalize_answer(x):
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


# ------------------------------------------------------------
# Cleaner question taxonomy
# ------------------------------------------------------------

def classify_question(q):
    q = " " + str(q).lower().strip() + " "

    if (
        " what color " in q
        or " what colour " in q
        or " what is the color " in q
        or " what is the colour " in q
        or " what color are " in q
        or " what colour are " in q
    ):
        return "color"

    if (
        " what material " in q
        or " what is the material " in q
        or " made of " in q
        or " made from " in q
    ):
        return "material"

    if (
        " what shape " in q
        or " what is the shape " in q
        or " what shape do " in q
    ):
        return "shape"

    if (
        " what is this " in q
        or " what is that " in q
        or " what kind of " in q
        or " what type of " in q
        or " what is the name "
        or " what is this called "
        or " what is that called "
    ):
        return "identity"

    return "other"


changed_correctness = []
changed_category = []

old_cat_counts = Counter()
new_cat_counts = Counter()

for r in rows:

    old_cat = str(r.get("category", "unknown"))
    new_cat = classify_question(r["question"])

    old_cat_counts[old_cat] += 1
    new_cat_counts[new_cat] += 1

    if old_cat != new_cat:
        changed_category.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "old_category": old_cat,
            "new_category": new_cat,
        })

    r["category_original"] = old_cat
    r["category"] = new_cat

    old_correct = bool(r["correct"])

    new_correct = (
        normalize_answer(r["answer"])
        == normalize_answer(r["ground_truth"])
    )

    if old_correct != new_correct:
        changed_correctness.append({
            "question_id": r["question_id"],
            "severity": r["severity"],
            "answer": r["answer"],
            "ground_truth": r["ground_truth"],
            "old_correct": old_correct,
            "new_correct": new_correct,
        })

    r["correct"] = new_correct


# ------------------------------------------------------------
# Recompute category EMVR
# ------------------------------------------------------------

by_q = defaultdict(list)

for r in rows:
    by_q[str(r["question_id"])].append(r)

category_stats = defaultdict(
    lambda: {"violations": 0, "pairs": 0}
)

for qid, traj in by_q.items():

    traj = sorted(
        traj,
        key=lambda x: float(x["severity"])
    )

    if len(traj) != 5:
        continue

    cat = traj[0]["category"]

    for a, b in zip(traj[:-1], traj[1:]):

        if float(b["c_seq"]) > float(a["c_seq"]):
            category_stats[cat]["violations"] += 1

        category_stats[cat]["pairs"] += 1


for cat, d in category_stats.items():
    d["emvr"] = (
        d["violations"] / d["pairs"]
        if d["pairs"] else None
    )


# ------------------------------------------------------------
# Save cleaned results
# ------------------------------------------------------------

clean_path = OUTDIR / "results_cleaned.json"

clean_path.write_text(
    json.dumps(rows, indent=2),
    encoding="utf-8"
)

report = {
    "rows": len(rows),
    "trajectories": len(by_q),
    "old_category_counts": dict(old_cat_counts),
    "new_category_counts": dict(new_cat_counts),
    "category_changes": len(changed_category),
    "correctness_changes": len(changed_correctness),
    "category_emvr": dict(category_stats),
}

(OUTDIR / "cleanup_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

(OUTDIR / "category_changes.json").write_text(
    json.dumps(changed_category, indent=2),
    encoding="utf-8"
)

(OUTDIR / "correctness_changes.json").write_text(
    json.dumps(changed_correctness, indent=2),
    encoding="utf-8"
)


print("=" * 72)
print("P17A DATASET / EVALUATOR CLEANUP")
print("=" * 72)

print("Rows:", len(rows))
print("Trajectories:", len(by_q))

print("\nOld category counts:")
for k, v in old_cat_counts.items():
    print(k, v)

print("\nNew category counts:")
for k, v in new_cat_counts.items():
    print(k, v)

print("\nCategory labels changed:", len(changed_category))
print("Correctness labels changed:", len(changed_correctness))

print("\nRecomputed category EMVR:")
for cat, d in category_stats.items():
    print(
        f"{cat}: "
        f"{d['violations']}/{d['pairs']} "
        f"EMVR={d['emvr']:.4f}"
    )

print("\nSaved:", OUTDIR)
print("P17A COMPLETE")
