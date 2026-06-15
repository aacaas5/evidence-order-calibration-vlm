import zipfile
from pathlib import Path

archives = [
    Path(r"data\gqa\archives\sceneGraphs.zip"),
    Path(r"data\gqa\archives\questions1.2.zip"),
]

for archive in archives:
    print("\n" + "=" * 72)
    print("ARCHIVE:", archive)
    print("=" * 72)

    with zipfile.ZipFile(archive, "r") as z:
        names = z.namelist()

        print("Number of files:", len(names))

        for name in names[:100]:
            print(name)

        if len(names) > 100:
            print(f"... {len(names) - 100} more files")
