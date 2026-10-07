import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


SOURCE_DIR = Path(
    r"data\processed\images"
)

OUTPUT_DIR = Path(
    r"data\processed\splits"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


classes = sorted(
    [
        folder.name
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]
)


print("Classes:")
print(classes)


for split in [
    "train",
    "val",
    "test"
]:

    for class_name in classes:

        directory = (
            OUTPUT_DIR
            / split
            / class_name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


for class_name in classes:

    source_class_dir = (
        SOURCE_DIR / class_name
    )

    files = sorted(
        [
            file
            for file in source_class_dir.iterdir()
            if file.suffix.lower() == ".png"
        ]
    )


    train_files, temp_files = train_test_split(
        files,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42
    )


    val_files, test_files = train_test_split(
        temp_files,
        test_size=(
            TEST_RATIO /
            (VAL_RATIO + TEST_RATIO)
        ),
        random_state=42
    )


    for file in train_files:

        shutil.copy2(
            file,
            OUTPUT_DIR
            / "train"
            / class_name
            / file.name
        )


    for file in val_files:

        shutil.copy2(
            file,
            OUTPUT_DIR
            / "val"
            / class_name
            / file.name
        )


    for file in test_files:

        shutil.copy2(
            file,
            OUTPUT_DIR
            / "test"
            / class_name
            / file.name
        )


    print(
        f"{class_name}: "
        f"Train={len(train_files)}, "
        f"Val={len(val_files)}, "
        f"Test={len(test_files)}"
    )


print("\nSplit complete.")

print(
    "Output:",
    OUTPUT_DIR
)