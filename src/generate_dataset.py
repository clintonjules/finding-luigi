import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import os
import shutil
import random
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# adapted from
# https://github.com/nathancooperjones/super-mario-64-ds-wanted-autoplay/blob/main/01_generate_data/generate-data.ipynb

CANVAS = (720, 720)
# CANVAS = (191, 255)

BASE_DIR = "character_images"

ALL_IMAGES = list(Path(BASE_DIR).glob("*.png"))


CHARACTER_BOX_COLOR_DICTIONARY = {
    "luigi": "#34eb43",
    "wario": "#ebdb34",
    "yoshi": "#cc1bae",
    "mario": "#cc1b1b",
}

COCO_CHARACTER_CATEGORIES_DICT = {
    "categories": [
        {"id": 1, "name": "mario"},
        {"id": 2, "name": "luigi"},
        {"id": 3, "name": "wario"},
        {"id": 4, "name": "yoshi"},
    ],
}


def create_wanted_minigame_screen(
    num_characters: int = 15,
    image_filename: Optional[str] = None,
    plot_aspect_ratio: Tuple[int, int] = CANVAS,
    display_plot: bool = True,
    display_boxes: bool = False,
    proximity_threshold: float = 5,
    zoom_multiplier: float = 1,
    pad_inches: float = 0.0,
) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    
    fig, ax = plt.subplots(figsize=(CANVAS[0] / 65, CANVAS[1] / 65))
    # fig, ax = plt.subplots()

    characters_chosen = list()
    used_coordinates_list = list()
    luigi_added = False  # Ensure only one Luigi

    ax.imshow(X=np.zeros(shape=plot_aspect_ratio + (3,)))

    for i in range(num_characters):
        while_loop_counter = 0

        while while_loop_counter <= 100:
            while_loop_counter += 1

            x = np.random.randint(low=0, high=plot_aspect_ratio[1])
            y = np.random.randint(low=0, high=plot_aspect_ratio[0])

            # avoid overlap with previously-plotted characters
            for used_x, used_y in used_coordinates_list:
                if abs(used_x - x) < (
                    proximity_threshold * (zoom_multiplier**2)
                ) and abs(used_y - y) < (proximity_threshold * (zoom_multiplier**2)):
                    # break out of sub-for loop
                    break
            else:
                # break out of while loop
                break

        if while_loop_counter > 100:
            # character can't fit without occlusion - stop generating
            break

        used_coordinates_list.append((x, y))

        if not luigi_added:
            character_chosen = "luigi"
            luigi_added = True
        else:
            character_chosen_path = Path(np.random.choice(ALL_IMAGES))
            character_chosen = character_chosen_path.stem
            # Ensure no additional Luigi
            while character_chosen == "luigi":
                character_chosen_path = Path(np.random.choice(ALL_IMAGES))
                character_chosen = character_chosen_path.stem

        characters_chosen.append(character_chosen)

        img = plt.imread(fname=Path(BASE_DIR) / f"{character_chosen}.png")
        img = OffsetImage(arr=img, zoom=zoom_multiplier)
        img.image.axes = ax

        bbox_kwargs = {
            "frameon": True,
            "bboxprops": {
                "edgecolor": CHARACTER_BOX_COLOR_DICTIONARY.get(character_chosen),
            },
            "pad": 0.1,
        }

        ab = AnnotationBbox(
            offsetbox=img,
            xy=(x, y),
            xycoords="data",
            **(bbox_kwargs if display_boxes else {}),
        )
        ab.patch._facecolor = (0, 0, 0, 0)

        if not display_boxes:
            # set the edges of the ``AnnotationBbox`` transparent
            ab.patch._edgecolor = (0, 0, 0, 0)

        ax.add_artist(a=ab)

    ax.set_axis_off()

    if display_plot:
        plt.show()

    if image_filename:
        Path(image_filename).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            fname=image_filename,
            bbox_inches="tight",
            pad_inches=pad_inches,
            facecolor=(0, 0, 0),  # black color code
        )

    plt.close(fig)

    return list(
        zip(
            characters_chosen,
            [
                (ab.patch._x, ab.patch._y, ab.patch._width, ab.patch._height)
                for ab in ax.artists
            ],
        )
    )


def generate_coco_annotations_and_images_dict(
    characters_bbox_list: List[Tuple[str, Tuple[float, float, float, float]]],
    starting_character_id: int,
    image_id: int,
    image_filename: str,
) -> List[Dict[str, Union[int, float, List[float]]]]:

    coco_annotations_list = list()
    coco_images_list = list()

    image_height, image_width, _ = plt.imread(fname=image_filename).shape

    for idx, (character_name, (bbox_x, bbox_y, bbox_w, bbox_h)) in enumerate(
        characters_bbox_list
    ):
        bbox_y = image_height - bbox_y - bbox_h

        # if entire character box is outside range, don't label it
        if (
            (bbox_x + bbox_w) <= 0  # too far left
            or bbox_x >= image_width  # too far right
            or (bbox_y + bbox_h) <= 0  # too far up
            or bbox_y >= image_height  # too far down
        ):
            continue

        adjusted_bbox_x = max(bbox_x, 0)
        adjusted_bbox_y = max(bbox_y, 0)
        adjusted_bbox_w = bbox_w
        adjusted_bbox_h = bbox_h

        if bbox_x <= 0:
            adjusted_bbox_w = bbox_x + adjusted_bbox_w
        elif (bbox_x + adjusted_bbox_w) >= image_width:
            adjusted_bbox_w = image_width - bbox_x

        if bbox_y <= 0:
            adjusted_bbox_h = bbox_y + adjusted_bbox_h
        elif (bbox_y + adjusted_bbox_h) >= image_height:
            adjusted_bbox_h = image_height - bbox_y

        coco_annotations_list.append(
            {
                "id": starting_character_id + idx,
                "image_id": image_id,
                "category_id": next(
                    d["id"]
                    for d in COCO_CHARACTER_CATEGORIES_DICT["categories"]
                    if d["name"] == character_name
                ),
                "segmentation": [],  # N/A
                "area": (adjusted_bbox_w * adjusted_bbox_h),
                # adjusting all four bbox coordinates to not go off screen
                "bbox": [
                    adjusted_bbox_x,
                    adjusted_bbox_y,
                    adjusted_bbox_w,
                    adjusted_bbox_h,
                ],
                "iscrowd": 0,  # N/A
            }
        )

    coco_images_list.append(
        {
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": str(image_filename),
        },
    )

    return coco_annotations_list, coco_images_list


def generate_object_detection_dataset(
    num_examples_to_generate: int = 20,
    num_characters_range: Tuple[int, int] = (4, 75),
    save_dir: str = "data/",
    zoom_multiplier_possible_values: Iterable[float] = [1],
    pad_inches_possible_values: Iterable[float] = [0.0],
    **kwargs,
) -> Dict[str, List[Dict[str, Union[int, float, List[float]]]]]:

    assert len(num_characters_range) == 2

    character_id_counter = 1
    coco_annotations_list = list()
    coco_images_list = list()

    image_directory = Path(save_dir) / "images"
    annotations_filename = Path(save_dir) / "annotations.json"

    for image_id in tqdm(range(1, num_examples_to_generate + 1)):
        num_characters = np.random.randint(
            low=num_characters_range[0],
            high=num_characters_range[1],
        )

        image_filename = Path(image_directory) / f"image_{image_id:05d}.png"

        characters_bbox_list = create_wanted_minigame_screen(
            num_characters=num_characters,
            image_filename=image_filename,
            plot_aspect_ratio=CANVAS,
            zoom_multiplier=np.random.choice(
                a=zoom_multiplier_possible_values,
            ),
            pad_inches=np.random.choice(
                a=pad_inches_possible_values,
            ),
            **kwargs,
        )

        (
            coco_annotations_list_updates,
            coco_images_list_updates,
        ) = generate_coco_annotations_and_images_dict(
            characters_bbox_list=characters_bbox_list,
            starting_character_id=character_id_counter,
            image_id=image_id,
            image_filename=image_filename,
        )

        coco_annotations_list += coco_annotations_list_updates
        coco_images_list += coco_images_list_updates

        character_id_counter += num_characters

    # format and save off annotations dictionary
    coco_annotations_dict = {
        **COCO_CHARACTER_CATEGORIES_DICT,
        "images": coco_images_list,
        "annotations": coco_annotations_list,
    }

    annotations_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(str(annotations_filename), "w") as fp:
        json.dump(coco_annotations_dict, fp)

    return coco_annotations_dict

def split_dataset(source_dir, output_dir, split_ratio):
    """
    Splits a dataset of images and labels into training and validation sets and creates an annotations.json file.

    Args:
        source_dir (str): Path to the source directory containing images and labels.
        output_dir (str): Path to the output directory for the split dataset.
        split_ratio (float): Proportion of the data to use for training (e.g., 0.8 for 80% training, 20% validation).
    """
    # Ensure source directory exists
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    # Define paths for images
    images_dir = source_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist.")

    # Get all image and label files
    files = list(images_dir.glob("*.*"))  # Includes both images and labels
    image_files = [file for file in files if file.suffix in [".png", ".jpg", ".jpeg"]]
    label_files = [file for file in files if file.suffix == ".txt"]

    # Match images and labels
    data = []
    for img in image_files:
        label = img.with_suffix(".txt")
        if label in label_files:
            data.append((str(img), str(label)))
        else:
            print(f"Warning: No label found for {img}")

    # Shuffle and split data
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # Create output directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    for folder in [train_dir, val_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # Move files to respective directories
    def move_files(data, dest_dir):
        for img_path, label_path in data:
            shutil.copy(img_path, dest_dir)
            shutil.copy(label_path, dest_dir)

    move_files(train_data, train_dir)
    move_files(val_data, val_dir)

    print(f"Dataset split completed!")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")

# coco_annotations_dict = generate_object_detection_dataset(
#     num_examples_to_generate=15,
#     num_characters_range=(4, 150),
#     save_dir=Path("data").resolve(),
#     display_plot=False,
#     display_boxes=False,
#     proximity_threshold=5,
#     pad_inches_possible_values=[-0.75],
# )

# split_dataset(
#     source_dir="data",  # Replace with your dataset directory
#     output_dir="data",  # Replace with the desired output directory
#     split_ratio=0.8  # 80% train, 20% validation
# )