import json
import re
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_PATH = Path(
    "data/gqa/metadata/val_balanced_questions.json"
)

SCENE_PATH = Path(
    "data/gqa/metadata/val_sceneGraphs.json"
)

OUT_MANIFEST = Path(
    "data/gqa/manifests/gqa_evidence_pilot_50.json"
)

OUT_STATS = Path(
    "results/raw/p4d_mapping_stats.json"
)

TARGET_TOTAL = 50


# ============================================================
# INITIAL FILTERING POLICY
# ============================================================
#
# We want relatively clean question types first.
#
# We are NOT saying other GQA questions are unusable.
# We are simply building a controlled pilot.
#
# ============================================================

ALLOWED_STRUCTURAL_TYPES = {
    "query",
    "verify",
    "choose",
    "compare",
    "logical",
}

PREFERRED_DETAILED_KEYWORDS = [
    "color",
    "exist",
    "position",
    "relation",
    "count",
    "number",
    "attribute",
    "category",
    "material",
    "shape",
    "size",
]


# ============================================================
# HELPERS
# ============================================================

def extract_object_ids_from_argument(argument):
    """
    Extract object IDs written inside parentheses.

    Example:
        "bird (329774)"
          ->
        ["329774"]

    Also handles cases with multiple IDs.
    """

    if not isinstance(argument, str):
        return []

    return re.findall(
        r"\((\d+)\)",
        argument,
    )


def extract_semantic_object_ids(semantic):
    """
    Walk through every semantic-program step and collect
    explicitly referenced GQA object IDs.
    """

    ids = []

    if not isinstance(semantic, list):
        return ids

    for step in semantic:

        if not isinstance(step, dict):
            continue

        argument = step.get(
            "argument",
            "",
        )

        ids.extend(
            extract_object_ids_from_argument(
                argument
            )
        )

    # Preserve order but remove duplicates
    seen = set()
    unique_ids = []

    for object_id in ids:

        if object_id not in seen:

            seen.add(object_id)
            unique_ids.append(
                object_id
            )

    return unique_ids


def xywh_to_xyxy(box):
    """
    Convert:

        [x, y, width, height]

    into:

        [x1, y1, x2, y2]
    """

    x, y, w, h = box

    return [
        x,
        y,
        x + w,
        y + h,
    ]


def classify_question(record):
    """
    Keep the original GQA types but also make a simple
    pilot-friendly category.

    This is only for experiment organization.
    """

    types = record.get(
        "types",
        {},
    )

    detailed = str(
        types.get(
            "detailed",
            "",
        )
    ).lower()

    semantic_type = str(
        types.get(
            "semantic",
            "",
        )
    ).lower()

    structural = str(
        types.get(
            "structural",
            "",
        )
    ).lower()

    question = str(
        record.get(
            "question",
            "",
        )
    ).lower()


    text = " ".join(
        [
            detailed,
            semantic_type,
            structural,
            question,
        ]
    )


    if "color" in text:
        category = "color"

    elif (
        "how many" in question
        or "count" in text
        or "number" in text
    ):
        category = "count"

    elif (
        question.startswith("is ")
        or question.startswith("are ")
        or "exist" in text
    ):
        category = "existence"

    elif any(
        keyword in text
        for keyword in [
            "left",
            "right",
            "above",
            "below",
            "behind",
            "front",
            "near",
            "position",
            "relation",
        ]
    ):
        category = "spatial"

    elif "material" in text:
        category = "material"

    elif "shape" in text:
        category = "shape"

    elif "size" in text:
        category = "size"

    elif (
        "category" in text
        or semantic_type == "cat"
    ):
        category = "category"

    else:
        category = "other"


    return {
        "pilot_category":
            category,

        "gqa_detailed":
            types.get(
                "detailed"
            ),

        "gqa_semantic":
            types.get(
                "semantic"
            ),

        "gqa_structural":
            types.get(
                "structural"
            ),
    }


def object_area_fraction(
    box,
    image_width,
    image_height,
):
    """
    Fraction of entire image occupied by the object box.

    Very tiny objects may be hard for the VLM.
    Extremely huge objects can also make intervention
    interpretation less clean.
    """

    x, y, w, h = box

    if (
        not image_width
        or not image_height
    ):
        return None

    image_area = (
        image_width
        * image_height
    )

    object_area = (
        max(0, w)
        * max(0, h)
    )

    return (
        object_area
        / image_area
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("PROJECT 3 - P4D GQA QUESTION -> EVIDENCE MAPPER")
print("=" * 78)

print("\n[1] Loading validation questions...")

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8",
) as f:

    questions = json.load(f)

print(
    "Questions:",
    len(questions),
)


print("\n[2] Loading validation scene graphs...")

with open(
    SCENE_PATH,
    "r",
    encoding="utf-8",
) as f:

    scene_graphs = json.load(f)

print(
    "Scene graphs:",
    len(scene_graphs),
)


# ============================================================
# BUILD CANDIDATE MAPPINGS
# ============================================================

print("\n[3] Mapping questions to explicit scene-graph objects...")

candidates = []

reject_reasons = Counter()


for question_id, record in questions.items():

    image_id = str(
        record.get(
            "imageId",
            "",
        )
    )


    if not image_id:

        reject_reasons[
            "missing_image_id"
        ] += 1

        continue


    graph = scene_graphs.get(
        image_id
    )


    if graph is None:

        reject_reasons[
            "scene_graph_missing"
        ] += 1

        continue


    semantic = record.get(
        "semantic",
        [],
    )


    object_ids = (
        extract_semantic_object_ids(
            semantic
        )
    )


    if not object_ids:

        reject_reasons[
            "no_explicit_object_id"
        ] += 1

        continue


    objects = graph.get(
        "objects",
        {}
    )


    valid_object_ids = [
        object_id
        for object_id in object_ids
        if object_id in objects
    ]


    if not valid_object_ids:

        reject_reasons[
            "referenced_objects_not_found"
        ] += 1

        continue


    # --------------------------------------------------------
    # For this first pilot we prefer ONE explicitly referenced
    # object. Multi-object questions will be handled later.
    # --------------------------------------------------------

    if len(valid_object_ids) != 1:

        reject_reasons[
            "multiple_critical_objects"
        ] += 1

        continue


    critical_id = (
        valid_object_ids[0]
    )

    obj = objects[
        critical_id
    ]


    required_box_fields = [
        "x",
        "y",
        "w",
        "h",
    ]


    if not all(
        field in obj
        for field in required_box_fields
    ):

        reject_reasons[
            "missing_bbox"
        ] += 1

        continue


    bbox_xywh = [
        int(obj["x"]),
        int(obj["y"]),
        int(obj["w"]),
        int(obj["h"]),
    ]


    if (
        bbox_xywh[2] <= 0
        or bbox_xywh[3] <= 0
    ):

        reject_reasons[
            "invalid_bbox"
        ] += 1

        continue


    width = graph.get(
        "width"
    )

    height = graph.get(
        "height"
    )


    area_fraction = (
        object_area_fraction(
            bbox_xywh,
            width,
            height,
        )
    )


    if area_fraction is None:

        reject_reasons[
            "missing_image_dimensions"
        ] += 1

        continue


    # --------------------------------------------------------
    # Pilot quality filter:
    #
    # Avoid very tiny objects because they are difficult to
    # manipulate and may not be visible to Qwen.
    #
    # Avoid nearly full-frame objects because removing them
    # destroys most of the image.
    # --------------------------------------------------------

    if area_fraction < 0.01:

        reject_reasons[
            "object_too_small"
        ] += 1

        continue


    if area_fraction > 0.65:

        reject_reasons[
            "object_too_large"
        ] += 1

        continue


    type_info = (
        classify_question(
            record
        )
    )


    structural = str(
        type_info[
            "gqa_structural"
        ]
    ).lower()


    if (
        structural
        and structural
        not in ALLOWED_STRUCTURAL_TYPES
    ):

        reject_reasons[
            "structural_type_filtered"
        ] += 1

        continue


    candidate = {

        "question_id":
            str(question_id),

        "image_id":
            image_id,

        "question":
            record.get(
                "question"
            ),

        "answer":
            record.get(
                "answer"
            ),

        "full_answer":
            record.get(
                "fullAnswer"
            ),

        "pilot_category":
            type_info[
                "pilot_category"
            ],

        "gqa_types": {
            "detailed":
                type_info[
                    "gqa_detailed"
                ],

            "semantic":
                type_info[
                    "gqa_semantic"
                ],

            "structural":
                type_info[
                    "gqa_structural"
                ],
        },

        "critical_object_ids": [
            critical_id
        ],

        "critical_objects": [
            {
                "object_id":
                    critical_id,

                "name":
                    obj.get(
                        "name"
                    ),

                "attributes":
                    obj.get(
                        "attributes",
                        [],
                    ),

                "bbox_xywh":
                    bbox_xywh,

                "bbox_xyxy":
                    xywh_to_xyxy(
                        bbox_xywh
                    ),

                "area_fraction":
                    round(
                        area_fraction,
                        6,
                    ),
            }
        ],

        "image_width":
            width,

        "image_height":
            height,

        "semantic_program":
            semantic,

        "mapping_method":
            "explicit_semantic_object_id",

        "mapping_confidence":
            "high",
    }


    candidates.append(
        candidate
    )


print(
    "High-confidence single-object candidates:",
    len(candidates),
)


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

category_counts = Counter(
    x["pilot_category"]
    for x in candidates
)


print("\n[4] Candidate category distribution:")

for category, count in (
    category_counts.most_common()
):

    print(
        f"   {category:<15} {count}"
    )


# ============================================================
# BUILD A BALANCED PILOT
# ============================================================

print(
    "\n[5] Selecting balanced 50-question pilot..."
)

preferred_order = [
    "color",
    "existence",
    "spatial",
    "count",
    "material",
    "shape",
    "size",
    "category",
    "other",
]


by_category = defaultdict(
    list
)


for candidate in candidates:

    by_category[
        candidate[
            "pilot_category"
        ]
    ].append(
        candidate
    )


# Deterministic ordering
for category in by_category:

    by_category[
        category
    ].sort(
        key=lambda x:
        x["question_id"]
    )


selected = []

seen_images = set()


# ------------------------------------------------------------
# Round-robin sampling.
#
# This helps stop one category from completely dominating.
#
# Also initially use one question per image whenever possible.
# ------------------------------------------------------------

while len(selected) < TARGET_TOTAL:

    added_this_round = False

    for category in preferred_order:

        pool = by_category[
            category
        ]

        chosen_index = None

        for i, candidate in enumerate(
            pool
        ):

            if (
                candidate["image_id"]
                not in seen_images
            ):

                chosen_index = i
                break


        if chosen_index is None:
            continue


        candidate = pool.pop(
            chosen_index
        )

        selected.append(
            candidate
        )

        seen_images.add(
            candidate["image_id"]
        )

        added_this_round = True


        if (
            len(selected)
            >= TARGET_TOTAL
        ):
            break


    if not added_this_round:
        break


# ------------------------------------------------------------
# If category balancing + unique images did not reach 50,
# fill from remaining candidates.
# ------------------------------------------------------------

if len(selected) < TARGET_TOTAL:

    selected_ids = {
        x["question_id"]
        for x in selected
    }

    for candidate in candidates:

        if (
            candidate["question_id"]
            in selected_ids
        ):
            continue

        selected.append(
            candidate
        )

        selected_ids.add(
            candidate["question_id"]
        )


        if (
            len(selected)
            >= TARGET_TOTAL
        ):
            break


# ============================================================
# SAVE MANIFEST
# ============================================================

OUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUT_MANIFEST,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# STATISTICS
# ============================================================

selected_categories = Counter(
    x["pilot_category"]
    for x in selected
)


stats = {

    "total_questions":
        len(questions),

    "high_confidence_candidates":
        len(candidates),

    "selected_pilot":
        len(selected),

    "candidate_category_counts":
        dict(category_counts),

    "selected_category_counts":
        dict(selected_categories),

    "reject_reasons":
        dict(reject_reasons),
}


with open(
    OUT_STATS,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        stats,
        f,
        indent=2,
    )


# ============================================================
# PRINT FIRST 10 MAPPINGS
# ============================================================

print("\n" + "=" * 78)
print("FIRST 10 SELECTED GQA-EVIDENCE MAPPINGS")
print("=" * 78)


for i, sample in enumerate(
    selected[:10],
    start=1,
):

    obj = (
        sample[
            "critical_objects"
        ][0]
    )

    print(
        f"\n[{i}]"
    )

    print(
        "Question ID:",
        sample[
            "question_id"
        ]
    )

    print(
        "Image ID:",
        sample[
            "image_id"
        ]
    )

    print(
        "Category:",
        sample[
            "pilot_category"
        ]
    )

    print(
        "Question:",
        sample[
            "question"
        ]
    )

    print(
        "Answer:",
        sample[
            "answer"
        ]
    )

    print(
        "Critical object:",
        obj[
            "name"
        ],
        f"({obj['object_id']})"
    )

    print(
        "Bounding box xywh:",
        obj[
            "bbox_xywh"
        ]
    )

    print(
        "Bounding box xyxy:",
        obj[
            "bbox_xyxy"
        ]
    )

    print(
        "Object area fraction:",
        obj[
            "area_fraction"
        ]
    )


print("\n" + "=" * 78)
print("SELECTED PILOT DISTRIBUTION")
print("=" * 78)

for category, count in (
    selected_categories.most_common()
):

    print(
        f"{category:<15} {count}"
    )


print("\n" + "=" * 78)
print("REJECTION STATISTICS")
print("=" * 78)

for reason, count in (
    reject_reasons.most_common()
):

    print(
        f"{reason:<32} {count}"
    )


print("\n" + "=" * 78)
print("P4D COMPLETE")
print("=" * 78)

print(
    "\nManifest:",
    OUT_MANIFEST
)

print(
    "Stats:",
    OUT_STATS
)

print(
    "\nSelected:",
    len(selected),
    "high-confidence pilot questions"
)

