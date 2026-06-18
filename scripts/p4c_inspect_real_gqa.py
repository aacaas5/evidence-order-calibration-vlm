import json
import zipfile
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

QUESTION_ZIP = Path(
    "data/gqa/archives/questions1.2.zip"
)

SCENE_ZIP = Path(
    "data/gqa/archives/sceneGraphs.zip"
)

META_DIR = Path(
    "data/gqa/metadata"
)

OUT_DIR = Path(
    "results/raw"
)

META_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 78)
print("PROJECT 3 - P4C REAL GQA METADATA INSPECTION")
print("=" * 78)


# ============================================================
# 1. VERIFY ARCHIVES
# ============================================================

print("\n[1] Verifying downloaded archives...")

for path in [
    QUESTION_ZIP,
    SCENE_ZIP,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing archive: {path}"
        )

    print(
        path,
        "->",
        round(
            path.stat().st_size / 1024**2,
            2,
        ),
        "MB",
    )


# ============================================================
# 2. INSPECT QUESTION ARCHIVE
# ============================================================

print("\n[2] Inspecting question archive...")

with zipfile.ZipFile(
    QUESTION_ZIP,
    "r",
) as z:

    question_names = z.namelist()


json_question_files = [
    n
    for n in question_names
    if n.lower().endswith(".json")
]


print(
    "JSON files in question archive:",
    len(json_question_files),
)


balanced_candidates = [
    n
    for n in json_question_files
    if "balanced" in n.lower()
]


print(
    "\nBalanced-question candidates:"
)

for name in balanced_candidates:
    print("   ", name)


# ============================================================
# 3. CHOOSE A SMALL USEFUL QUESTION FILE
# ============================================================

# Prefer validation balanced questions because they are usually
# smaller and ideal for pipeline development.

preferred_question = None

priority_terms = [
    "val_balanced_questions",
    "val_balanced",
    "train_balanced_questions",
    "train_balanced",
]


for term in priority_terms:

    matches = [
        n
        for n in balanced_candidates
        if term in n.lower()
    ]

    if matches:
        preferred_question = matches[0]
        break


if preferred_question is None:

    if balanced_candidates:
        preferred_question = balanced_candidates[0]

    else:
        raise RuntimeError(
            "No balanced GQA question JSON found."
        )


print(
    "\nChosen question file:"
)

print(
    preferred_question
)


# ============================================================
# 4. EXTRACT CHOSEN QUESTION FILE
# ============================================================

print("\n[3] Extracting chosen question metadata...")

question_output = (
    META_DIR /
    Path(preferred_question).name
)


with zipfile.ZipFile(
    QUESTION_ZIP,
    "r",
) as z:

    with z.open(
        preferred_question
    ) as source:

        with open(
            question_output,
            "wb",
        ) as target:

            while True:

                chunk = source.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                target.write(
                    chunk
                )


print(
    "Saved:",
    question_output
)


# ============================================================
# 5. LOAD QUESTIONS
# ============================================================

print("\n[4] Loading question JSON...")

with open(
    question_output,
    "r",
    encoding="utf-8",
) as f:

    questions = json.load(f)


print(
    "Top-level Python type:",
    type(questions).__name__,
)


if isinstance(
    questions,
    dict,
):

    print(
        "Number of questions:",
        len(questions),
    )

    question_id = next(
        iter(questions)
    )

    question_record = (
        questions[question_id]
    )

elif isinstance(
    questions,
    list,
):

    print(
        "Number of questions:",
        len(questions),
    )

    question_record = questions[0]

    question_id = str(
        question_record.get(
            "questionId",
            question_record.get(
                "question_id",
                "unknown",
            ),
        )
    )

else:
    raise RuntimeError(
        "Unexpected question JSON structure."
    )


# ============================================================
# 6. PRINT ONE REAL QUESTION
# ============================================================

print("\n" + "=" * 78)
print("ONE REAL GQA QUESTION")
print("=" * 78)

print(
    "Question ID:",
    question_id,
)

print(
    "\nAvailable fields:"
)

for key in question_record.keys():
    print("   ", key)


print(
    "\nQuestion:",
    question_record.get(
        "question",
        "<missing>",
    ),
)

print(
    "Answer:",
    question_record.get(
        "answer",
        "<missing>",
    ),
)

image_id = (
    question_record.get(
        "imageId"
    )
    or question_record.get(
        "image_id"
    )
)


print(
    "Image ID:",
    image_id,
)


# ============================================================
# 7. INSPECT QUESTION SEMANTICS / PROGRAM
# ============================================================

print("\n" + "=" * 78)
print("QUESTION SEMANTIC STRUCTURE")
print("=" * 78)


semantic_keys = [
    "semantic",
    "program",
    "types",
    "groups",
    "annotations",
    "entailed",
]


for key in semantic_keys:

    if key in question_record:

        print(
            f"\n{key}:"
        )

        value = question_record[key]

        print(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
            )[:8000]
        )


# ============================================================
# 8. INSPECT SCENE GRAPH ARCHIVE
# ============================================================

print("\n[5] Inspecting scene graph archive...")

with zipfile.ZipFile(
    SCENE_ZIP,
    "r",
) as z:

    scene_names = z.namelist()


scene_json_files = [
    n
    for n in scene_names
    if n.lower().endswith(".json")
]


print(
    "Scene graph JSON files:"
)

for name in scene_json_files:
    print("   ", name)


# ============================================================
# 9. EXTRACT ALL SMALL SCENE GRAPH JSON FILES
# ============================================================

print(
    "\n[6] Extracting scene graph JSON files..."
)

scene_outputs = []


with zipfile.ZipFile(
    SCENE_ZIP,
    "r",
) as z:

    for member in scene_json_files:

        out_path = (
            META_DIR /
            Path(member).name
        )

        if not out_path.exists():

            print(
                "Extracting:",
                member,
            )

            with z.open(
                member
            ) as source:

                with open(
                    out_path,
                    "wb",
                ) as target:

                    while True:

                        chunk = source.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        target.write(
                            chunk
                        )

        scene_outputs.append(
            out_path
        )


# ============================================================
# 10. FIND MATCHING IMAGE SCENE GRAPH
# ============================================================

print("\n[7] Searching for matching scene graph...")

matching_graph = None
matching_source = None


for scene_path in scene_outputs:

    print(
        "Checking:",
        scene_path.name
    )

    with open(
        scene_path,
        "r",
        encoding="utf-8",
    ) as f:

        graphs = json.load(f)


    if (
        image_id is not None
        and str(image_id) in graphs
    ):

        matching_graph = graphs[
            str(image_id)
        ]

        matching_source = (
            scene_path.name
        )

        break


if matching_graph is None:

    print(
        "\nWARNING:"
    )

    print(
        "The first question's image was not found "
        "in the extracted scene graph splits."
    )

    print(
        "We will choose another question whose "
        "scene graph is available."
    )


    # Build a set of available scene graph image IDs.

    available_graphs = {}

    for scene_path in scene_outputs:

        with open(
            scene_path,
            "r",
            encoding="utf-8",
        ) as f:

            graphs = json.load(f)

        for img_id, graph in graphs.items():

            if img_id not in available_graphs:

                available_graphs[
                    img_id
                ] = (
                    graph,
                    scene_path.name,
                )


    found = False

    if isinstance(
        questions,
        dict,
    ):

        iterator = (
            questions.items()
        )

    else:

        iterator = enumerate(
            questions
        )


    for qid, record in iterator:

        candidate_image = (
            record.get(
                "imageId"
            )
            or record.get(
                "image_id"
            )
        )

        if (
            candidate_image is not None
            and str(candidate_image)
            in available_graphs
        ):

            question_id = str(qid)
            question_record = record
            image_id = str(
                candidate_image
            )

            matching_graph, matching_source = (
                available_graphs[
                    image_id
                ]
            )

            found = True
            break


    if not found:

        raise RuntimeError(
            "Could not find any question with "
            "a matching scene graph."
        )


# ============================================================
# 11. PRINT MATCHED QUESTION AGAIN
# ============================================================

print("\n" + "=" * 78)
print("MATCHED REAL GQA SAMPLE")
print("=" * 78)

print(
    "Question ID:",
    question_id,
)

print(
    "Image ID:",
    image_id,
)

print(
    "Question:",
    question_record.get(
        "question",
        "<missing>",
    ),
)

print(
    "Answer:",
    question_record.get(
        "answer",
        "<missing>",
    ),
)

print(
    "Scene graph source:",
    matching_source,
)


# ============================================================
# 12. PRINT SCENE GRAPH BASIC INFO
# ============================================================

print("\n" + "=" * 78)
print("SCENE GRAPH")
print("=" * 78)


print(
    "Graph fields:",
    list(
        matching_graph.keys()
    ),
)


image_width = matching_graph.get(
    "width"
)

image_height = matching_graph.get(
    "height"
)


print(
    "Image width:",
    image_width,
)

print(
    "Image height:",
    image_height,
)


objects = matching_graph.get(
    "objects",
    {}
)


print(
    "Number of objects:",
    len(objects),
)


# ============================================================
# 13. PRINT FIRST OBJECTS + BOUNDING BOXES
# ============================================================

print("\nObjects in image:")

object_summary = []


for i, (
    object_id,
    obj,
) in enumerate(
    objects.items()
):

    name = obj.get(
        "name",
        "<unknown>",
    )

    x = obj.get("x")
    y = obj.get("y")
    w = obj.get("w")
    h = obj.get("h")

    attributes = obj.get(
        "attributes",
        [],
    )

    bbox = [
        x,
        y,
        w,
        h,
    ]

    object_summary.append(
        {
            "object_id":
                object_id,

            "name":
                name,

            "bbox_xywh":
                bbox,

            "attributes":
                attributes,
        }
    )


    print(
        f"\nObject {object_id}"
    )

    print(
        "   name:",
        name,
    )

    print(
        "   bbox [x,y,w,h]:",
        bbox,
    )

    if attributes:

        print(
            "   attributes:",
            attributes[:10],
        )


    if i >= 19:

        print(
            "\n... remaining objects omitted"
        )

        break


# ============================================================
# 14. SHOW RELATIONS FOR FIRST FEW OBJECTS
# ============================================================

print("\n" + "=" * 78)
print("OBJECT RELATIONS")
print("=" * 78)


shown = 0

for object_id, obj in objects.items():

    relations = obj.get(
        "relations",
        [],
    )

    if not relations:
        continue


    print(
        f"\n{object_id} ({obj.get('name')})"
    )

    for relation in relations[:10]:

        print(
            "   ",
            relation,
        )


    shown += 1

    if shown >= 5:
        break


# ============================================================
# 15. SAVE HUMAN-READABLE INSPECTION FILE
# ============================================================

inspection = {
    "question_id":
        str(question_id),

    "image_id":
        str(image_id),

    "question":
        question_record.get(
            "question"
        ),

    "answer":
        question_record.get(
            "answer"
        ),

    "question_record":
        question_record,

    "scene_graph_source":
        matching_source,

    "image_width":
        image_width,

    "image_height":
        image_height,

    "objects":
        object_summary,
}


inspection_path = (
    OUT_DIR /
    "p4c_real_gqa_sample.json"
)


with open(
    inspection_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        inspection,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 78)
print("P4C COMPLETE")
print("=" * 78)

print(
    "Inspection saved:",
    inspection_path,
)

print(
    "\nNext step:"
)

print(
    "Map this question's semantic program "
    "to the scene-graph object(s)."
)
