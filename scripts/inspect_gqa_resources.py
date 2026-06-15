from pathlib import Path
import urllib.request

print("=" * 72)
print("PROJECT 3 - P4A GQA RESOURCE INSPECTOR")
print("=" * 72)

resources = {
    "images": (
        "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    ),
    "scene_graphs": (
        "https://downloads.cs.stanford.edu/nlp/data/gqa/"
        "sceneGraphs.zip"
    ),
    "questions": (
        "https://downloads.cs.stanford.edu/nlp/data/gqa/"
        "questions1.2.zip"
    ),
}

print("\nWe are checking the GQA resources WITHOUT downloading them.\n")

for name, url in resources.items():

    print("-" * 72)
    print("Resource:", name)
    print("URL:", url)

    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:

            size = response.headers.get(
                "Content-Length"
            )

            content_type = response.headers.get(
                "Content-Type"
            )

            print(
                "HTTP status:",
                response.status,
            )

            print(
                "Content type:",
                content_type,
            )

            if size is not None:

                size_gb = (
                    int(size) /
                    1024**3
                )

                print(
                    "Reported size:",
                    round(size_gb, 3),
                    "GB",
                )

            else:
                print(
                    "Reported size: unavailable"
                )

    except Exception as exc:

        print(
            "HEAD request failed:",
            type(exc).__name__,
            str(exc),
        )

print("\n" + "=" * 72)
print("P4A COMPLETE")
print("=" * 72)

print(
    "\nNo GQA dataset files were downloaded."
)

print(
    "This step only checks which official resources are reachable."
)
