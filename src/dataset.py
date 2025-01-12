import numpy as np
from PIL import Image
import random
import os
import math
import pandas as pd
import itertools
from tqdm import tqdm

# Both methods to count the number of non-zero alpha pixels in an image seem off
# The second one is more accurate but still not perfect
# Sticking to declaring indepentently within fucntion for now
def count_non_zero_alpha_pixels(image: Image.Image) -> int:
    """
    Count the number of alpha channel pixels above 0 in an image.

    Args:
        image (Image.Image): The input image.

    Returns:
        int: Number of alpha channel pixels above 0.
    """
    
    # Extract the alpha channel
    alpha_channel = np.array(image)[:, :, 3]
    
    # Count the number of pixels with alpha value above 0
    return np.count_nonzero(alpha_channel > 0)

def get_non_transparent_pixels(image: Image.Image, height: int = None, width: int = None) -> set[tuple[int, int]]:
    image_array = np.array(image)
    
    if height is None:
        height = image.height
        
    if width is None:
        width = image.width
    
    return set(
        (x, y) for y in range(height) for x in range(width)
        if image_array[y, x, 3] > 0  # Check alpha channel for non-transparent pixels
    )
    

def calculate_bounding_box(visible_pixels):
    """
    Calculate the bounding box for Luigi based on visible pixels.

    Args:
        visible_pixels (list): List of (x, y) coordinates for Luigi's visible pixels.

    Returns:
        tuple: Bounding box coordinates (x_min, y_min, x_max, y_max).
    """

    x_coords, y_coords = zip(*visible_pixels)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return x_min, y_min, x_max, y_max

# Doesnt actually make permuatiions, forgot that functionality, woops!
def generate_permutations_excluding_overlay(input_filepaths, excluded_overlay):
    """
    Generate permutations where one file acts as the base, and others are overlays,
    excluding a specific file from being an overlay.

    Args:
        input_filepaths (list[str]): List of file paths.
        excluded_overlay (str): Filepath to exclude from being used as an overlay.

    Returns:
        list[tuple[str, list[str]]]: List of tuples where each tuple contains:
            - The base image filepath.
            - A list of overlay image filepaths.
    """
    # Filter out the excluded overlay from the input filepaths
    valid_overlays = [filepath for filepath in input_filepaths if filepath != excluded_overlay]

    # Generate permutations
    permutations_list = []
    for base in input_filepaths:
        if base == excluded_overlay:
            # Skip the excluded file from being an overlay
            overlays = valid_overlays
        else:
            # All files except the current base and the excluded overlay
            overlays = [filepath for filepath in input_filepaths if filepath != base and filepath != excluded_overlay]

        permutations_list.append((base, overlays))

    return permutations_list

def random_overlay(base_image_path: str, overlay_image_path: str, output_image_path: str = "output.png") -> tuple[Image.Image, int, int]:
    # Open the base image
    base_image = Image.open(base_image_path).convert("RGBA")
    base_width, base_height = base_image.size

    # Open the overlay image
    overlay_image = Image.open(overlay_image_path).convert("RGBA")
    overlay_width, overlay_height = overlay_image.size

    # Randomly choose top-left coordinates for the overlay
    x_offset = random.randint(-overlay_width, base_width)
    y_offset = random.randint(-overlay_height, base_height)

    # Create a new image to combine the images
    combined_image = Image.new("RGBA", (base_width, base_height), (0, 0, 0, 0))
    combined_image.paste(base_image, (0, 0))
    
    overlay_image_area = 0

        # Identify all non-transparent pixels in the base image
    base_image_array = np.array(base_image)
    valid_base_pixels = set(
        (x, y) for y in range(base_height) for x in range(base_width)
        if base_image_array[y, x, 3] > 0  # Check alpha channel for non-transparent pixels
    )
    base_image_area = len(valid_base_pixels)
    
    # Loop through the overlay pixels and place them on the base
    for x in range(overlay_width):
        for y in range(overlay_height):
            # Calculate the position on the base image
            base_x = x_offset + x
            base_y = y_offset + y

            # Only paste if within bounds of the base image
            if 0 <= base_x < base_width and 0 <= base_y < base_height:
                overlay_pixel = overlay_image.getpixel((x, y))
                if overlay_pixel[3] > 0:  # Only consider non-transparent pixels
                    combined_image.putpixel((base_x, base_y), overlay_pixel)
                    overlay_image_area += 1

    overlapping_area = overlay_image_area / base_image_area

    # Save the resulting image
    combined_image.save(output_image_path, format="PNG")

    return combined_image, x_offset, y_offset, overlay_image_area, base_image_area, overlapping_area


def random_overlay_multiple(base_image_path: str, overlay_image_paths: list, output_image_path: str = "output.png", 
                            max_overlap_percentage: float = 0.5, max_retries = float("inf"), 
                            max_base_coverage: float = 0.8, hard_cap: bool = True, **kwargs) -> Image.Image:
    """
    Overlay multiple images randomly on top of a base image while limiting overlap percentage and base coverage.

    Args:
        base_image_path (str): Path to the base image.
        overlay_image_paths (list): List of paths to overlay images.
        output_image_path (str): Path to save the output image.
        max_overlap_percentage (float): Maximum allowed overlap percentage for each overlay.
        max_retries (int): Maximum number of retries to fit an overlay within the threshold.
        max_base_coverage (float): Maximum allowed base coverage percentage.
    """
    # Open the base image
    base_image = Image.open(base_image_path).convert("RGBA")
    base_width, base_height = base_image.size

    # Identify all non-transparent pixels in the base image
    base_image_array = np.array(base_image)
    valid_base_pixels = set(
        (x, y) for y in range(base_height) for x in range(base_width)
        if base_image_array[y, x, 3] > 0  # Check alpha channel for non-transparent pixels
    )
    base_image_area = len(valid_base_pixels)
    
    retries = -1
    base_coverage_percentage = 0

    pbar = tqdm(total=max_retries)
    while retries < max_retries:
        retries += 1
        print(f"Retries left: {max_retries - retries}")
        
        # Reset the combined image and base pixels covered
        combined_image = Image.new("RGBA", (base_width, base_height), (0, 0, 0, 0))
        combined_image.paste(base_image, (0, 0))
        base_pixels_covered = set()  # Store coordinates of covered base pixels
        overlapping_values = {}
        
        for overlay_image_path in overlay_image_paths:
            overlay_placed = False
            overlay_retries = -1

            while not overlay_placed and overlay_retries < max_retries:
                overlay_retries += 1
                
                visible_luigi_pixels = valid_base_pixels.copy()

                # Open the overlay image
                overlay_image = Image.open(overlay_image_path).convert("RGBA")
                overlay_width, overlay_height = overlay_image.size

                # Randomly choose top-left coordinates for the overlay
                x_offset = random.randint(-overlay_width, base_width)
                y_offset = random.randint(-overlay_height, base_height)

                overlay_image_area = 0
                overlap_area = 0
                current_overlay_covered_pixels = set()

                # Loop through the overlay pixels and calculate overlap
                for x in range(overlay_width):
                    for y in range(overlay_height):
                        # Calculate the position on the base image
                        base_x = x_offset + x
                        base_y = y_offset + y

                        # Only consider pixels within bounds of the base image
                        if 0 <= base_x < base_width and 0 <= base_y < base_height:
                            overlay_pixel = overlay_image.getpixel((x, y))
                            base_pixel = combined_image.getpixel((base_x, base_y))

                            if overlay_pixel[3] > 0:  # Non-transparent overlay pixel
                                overlay_image_area += 1

                                # Count overlap only for non-transparent base pixels
                                if base_pixel[3] > 0:
                                    overlap_area += 1
                                    current_overlay_covered_pixels.add((base_x, base_y))

                # Calculate overlap percentage
                overlap_percentage = overlap_area / base_image_area
                
                # Update Luigi's visible pixels by removing any overwritten by this overlay
                visible_luigi_pixels -= current_overlay_covered_pixels
                
                # Check if the overlap is within the allowed threshold
                if overlap_percentage <= max_overlap_percentage or not hard_cap:
                    # Apply the overlay to the combined image
                    for x in range(overlay_width):
                        for y in range(overlay_height):
                            base_x = x_offset + x
                            base_y = y_offset + y

                            if 0 <= base_x < base_width and 0 <= base_y < base_height:
                                overlay_pixel = overlay_image.getpixel((x, y))
                                if overlay_pixel[3] > 0:
                                    combined_image.putpixel((base_x, base_y), overlay_pixel)
                                    base_pixels_covered.add((base_x, base_y))

                    # Save overlay details and mark as placed
                    overlapping_values[overlay_image_path] = {
                        "overlay_image_area": overlay_image_area,
                        "x_offset": x_offset,
                        "y_offset": y_offset,
                        "overlap_area": overlap_area,
                        "overlap_percentage": overlap_percentage,
                        "retries": overlay_retries
                    }
                    overlay_placed = True

            if (not overlay_placed) and hard_cap:
                print(f"Warning: Failed to place overlay {overlay_image_path} within {max_retries} retries.")

        # Calculate the total base coverage percentage
        valid_base_pixels_covered = base_pixels_covered.intersection(valid_base_pixels)
        base_coverage_percentage = len(valid_base_pixels_covered) / base_image_area
        
        pbar.update(1)

        # Break if base coverage is within the allowed limit
        if base_coverage_percentage <= max_base_coverage or not hard_cap:
            break
    
    pbar.close()
    
    if (base_coverage_percentage > max_base_coverage) and hard_cap:
        print(f"Warning: Failed to keep base coverage within {max_base_coverage * 100}% after {retries} retries.")

    # Save the resulting image
    combined_image.save(output_image_path, format="PNG")

    return {"visible_luigi_pixels": visible_luigi_pixels, "base_image_area": base_image_area, "base_coverage_percentage": base_coverage_percentage, "overlapping_values": overlapping_values}


def generate_random_image_subset_classify(base_image_path: str, overlay_image_paths: str, output_dir: str, num_samples: int=10, max_overlap_range: tuple=(0.0, 0.6), max_base_coverage_range: tuple=(0.0, 0.8), **kwargs):
    """
    Generate images with overlays applied to a base image using random_overlay_multiple.

    Args:
        base_image_path (str): Path to the base image.
        overlay_image_paths (list): List of overlay image paths.
        output_dir (str): Directory to save the generated images.
        num_samples (int): Number of samples to generate.
        **kwargs: Additional keyword arguments for random_overlay_multiple.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Generate output image path
        output_file_name = f"{os.path.basename(base_image_path).split('.')[0]}_sample_{i}.png"
        output_image_path = os.path.join(output_dir, output_file_name)
        
        # Randomly select values within the specified ranges
        max_overlap_percentage = random.uniform(*max_overlap_range)
        max_base_coverage = random.uniform(*max_base_coverage_range)

        # Call random_overlay_multiple with filtered arguments
        random_overlay_multiple(
            base_image_path=base_image_path,
            overlay_image_paths=overlay_image_paths,
            output_image_path=output_image_path,
            max_overlap_percentage=max_overlap_percentage, 
            max_base_coverage=max_base_coverage,
            **kwargs
        )


def generate_random_image_subset_bbox(base_image_path: str, overlay_image_paths: str, output_dir: str, num_samples: int=1, max_overlap_range: tuple=(0.0, 0.7), max_base_coverage_range: tuple=(0.0, 0.9), label='luigi', **kwargs):
    """
    Generate images with overlays applied to a base image using random_overlay_multiple.

    Args:
        base_image_path (str): Path to the base image.
        overlay_image_paths (list): List of overlay image paths.
        output_dir (str): Directory to save the generated images.
        num_samples (int): Number of samples to generate.
        **kwargs: Additional keyword arguments for random_overlay_multiple.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir + '/luigi', exist_ok=True)
    
    overlay_generator = generate_random_filepaths(overlay_image_paths, num_samples)

    for i in tqdm(range(num_samples)):
        print(f"Working on Sample {i}")
        
        # Generate output image path
        output_file_name = f"{os.path.basename(base_image_path).split('.')[0]}_sample_{i}.png"
        output_image_path = os.path.join(output_dir + '/luigi', output_file_name)
        output_bbox_path = os.path.join(output_dir, "bounding_boxes.csv")
        
        # Randomly select values within the specified ranges
        max_overlap_percentage = random.uniform(*max_overlap_range)
        max_base_coverage = random.uniform(*max_base_coverage_range)
        
        
        # Call random_overlay_multiple with filtered arguments
        random_image = random_overlay_multiple(
            base_image_path=base_image_path,
            overlay_image_paths=next(overlay_generator),
            output_image_path=output_image_path,
            max_overlap_percentage=max_overlap_percentage, 
            max_base_coverage=max_base_coverage,
            **kwargs
        )

        bbox = calculate_bounding_box(random_image["visible_luigi_pixels"])
        
        save_image_and_bounding_box(bbox, output_image_path, output_bbox_path, label)
    

def generate_classification_dataset(image_paths: str, excluded_overlay: str, num_samples: int = 100000, output_dir: str = "data", **kwargs):
    luigi_dir = os.path.join(output_dir, "luigi")
    not_luigi_dir = os.path.join(output_dir, "not_luigi")

    os.makedirs(luigi_dir, exist_ok=True)
    os.makedirs(not_luigi_dir, exist_ok=True)
    
    permutations_list = generate_permutations_excluding_overlay(image_paths, excluded_overlay)
    
    for base, overlays in permutations_list:
        if "luigi" in base:
            generate_random_image_subset_classify(base, overlays, luigi_dir, num_samples=math.floor(num_samples * 1.5), **kwargs)
        else:
            generate_random_image_subset_classify(base, overlays, not_luigi_dir, num_samples=num_samples//2, **kwargs)
            

def save_image_and_bounding_box(bounding_box, output_image_path, metadata_csv_path, label):
    # Prepare metadata
    data = {
        "image_path": output_image_path,
        "x_min": bounding_box[0],
        "y_min": bounding_box[1],
        "x_max": bounding_box[2],
        "y_max": bounding_box[3],
        "label": label,
    }

    # Append metadata to CSV
    if not os.path.exists(metadata_csv_path):
        pd.DataFrame([data]).to_csv(metadata_csv_path, index=False)
    else:
        pd.DataFrame([data]).to_csv(metadata_csv_path, mode='a', header=False, index=False)
    
    
def generate_bbox_dataset(base_image_path, overlay_image_paths, num_samples = 10, output_dir: str = "data", **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    
    overlay_permutations = list(itertools.permutations(overlay_image_paths))
    
    generate_random_image_subset_bbox(base_image_path, random.choice(overlay_permutations), output_dir, num_samples, **kwargs)    
    
    
def generate_random_filepaths(filepaths: list, num_samples: int, max_length: int = 12):
    all_random_lists = []
    
    for _ in range(num_samples):
        # Generate a random length for this sample
        bounds = random.randint(1, max_length)
        
        # Create a randomized list of filepaths
        random_list = [random.choice(filepaths) for _ in range(bounds)]
        random.shuffle(random_list)  # Shuffle the list
        
        yield random_list
        
        
def generate_background_dataset(num_samples=10, max_overlap_range: tuple=(0.0, 0.6), max_base_coverage_range: tuple=(0.0, 0.8), **kwargs):
    background_images = ["character_images/wario.png", "character_images/mario.png", "character_images/yoshi.png"]
    output_dir = "data/background"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    overlay_generator = generate_random_filepaths(background_images, num_samples)
    
    for i in tqdm(range(num_samples)):
        background = random.choice(background_images)

        # Generate output image path
        output_file_name = f"{os.path.basename(background).split('.')[0]}_sample_{i}.png"
        output_image_path = os.path.join(output_dir, output_file_name)
        
        # Randomly select values within the specified ranges
        max_overlap_percentage = random.uniform(*max_overlap_range)
        max_base_coverage = random.uniform(*max_base_coverage_range)

        # Call random_overlay_multiple with filtered arguments
        random_overlay_multiple(
            base_image_path=background,
            overlay_image_paths=next(overlay_generator),
            output_image_path=output_image_path,
            max_overlap_percentage=max_overlap_percentage, 
            max_base_coverage=max_base_coverage,
            max_retries=10,
            hard_cap=False,
            **kwargs
        )
        
        

# I hate this so much
# Here's some more broken code here
import csv
import random
from PIL import Image, ImageDraw
import os
import numpy as np


# Function to calculate bounding box for visible (non-transparent) pixels
def get_visible_bounding_box(image: Image.Image, x_offset: int, y_offset: int):
    image_array = np.array(image)
    alpha_channel = image_array[:, :, 3]
    non_transparent_pixels = np.argwhere(alpha_channel > 0)
    if non_transparent_pixels.size == 0:
        return None  # No visible pixels
    y_min, x_min = non_transparent_pixels.min(axis=0)
    y_max, x_max = non_transparent_pixels.max(axis=0)
    return x_min + x_offset, y_min + y_offset, x_max + x_offset, y_max + y_offset


# Function to crop an image to fit within canvas bounds
def crop_to_canvas(image, x, y, canvas_width, canvas_height):
    img_width, img_height = image.size

    # Calculate cropping box
    left = max(0, x)
    top = max(0, y)
    right = min(canvas_width, x + img_width)
    bottom = min(canvas_height, y + img_height)

    # Crop the image
    cropped_image = image.crop((left - x, top - y, right - x, bottom - y))

    # Return the cropped image and adjusted coordinates
    return cropped_image, (left, top)


# Function to update visibility mask after placing a character
def update_visibility_mask(visibility_mask, character_mask, x, y):
    canvas_height, canvas_width = visibility_mask.shape
    char_height, char_width = character_mask.shape

    # Calculate valid region of overlap
    top = max(0, y)
    left = max(0, x)
    bottom = min(canvas_height, y + char_height)
    right = min(canvas_width, x + char_width)

    char_top = top - y
    char_left = left - x
    char_bottom = bottom - y
    char_right = right - x

    # Update visibility mask
    visibility_mask[top:bottom, left:right] |= character_mask[char_top:char_bottom, char_left:char_right]


# Function to calculate the visibility of Luigi based on the visibility mask
def calculate_luigi_visibility(visibility_mask, luigi_mask, x, y):
    canvas_height, canvas_width = visibility_mask.shape
    luigi_height, luigi_width = luigi_mask.shape

    # Calculate valid region of overlap
    top = max(0, y)
    left = max(0, x)
    bottom = min(canvas_height, y + luigi_height)
    right = min(canvas_width, x + luigi_width)

    luigi_top = top - y
    luigi_left = left - x
    luigi_bottom = bottom - y
    luigi_right = right - x

    # Calculate visibility
    visible_area = (
        luigi_mask[luigi_top:luigi_bottom, luigi_left:luigi_right]
        & ~visibility_mask[top:bottom, left:right]
    )
    return np.count_nonzero(visible_area) / np.count_nonzero(luigi_mask)


# Function to generate 720x720 image samples and save bounding box data to CSV
def generate_image_samples_with_bboxes(
    output_folder,
    csv_path,
    base_image_size=(720, 720),
    num_samples=10,
    min_other_characters=5,
    max_other_characters=15,
    min_coverage=0.2,
    max_coverage=0.8,
    draw_bounding_box=False,
):
    luigi_folder = os.path.join(output_folder, "luigi")
    background_folder = os.path.join(output_folder, "background")
    os.makedirs(luigi_folder, exist_ok=True)
    os.makedirs(background_folder, exist_ok=True)

    character_paths = {
        "mario": "character_images/mario.png",
        "luigi": "character_images/luigi.png",
        "wario": "character_images/wario.png",
        "yoshi": "character_images/yoshi.png",
    }
    characters = {}
    for name, path in character_paths.items():
        img = Image.open(path).convert("RGBA")
        img.thumbnail((round(img.width * 0.1), round(img.height * 0.1)))
        characters[name] = img

    luigi_image = characters.pop("luigi")
    luigi_mask = np.array(luigi_image)[:, :, 3] > 0  # Non-transparent pixels

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_id", "character", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])

        for sample_idx in range(num_samples):
            canvas_with_luigi = Image.new("RGBA", base_image_size, (0, 0, 0, 255))
            canvas_without_luigi = Image.new("RGBA", base_image_size, (0, 0, 0, 255))
            visibility_mask = np.zeros(base_image_size[::-1], dtype=bool)  # Visibility mask
            luigi_bbox = None

            # Place Luigi first to ensure visibility
            while True:
                luigi_x = random.randint(-luigi_image.width, base_image_size[0])
                luigi_y = random.randint(-luigi_image.height, base_image_size[1])
                luigi_coverage = calculate_luigi_visibility(
                    visibility_mask, luigi_mask, luigi_x, luigi_y
                )
                if min_coverage <= luigi_coverage <= max_coverage:
                    break

            cropped_luigi, (cropped_x, cropped_y) = crop_to_canvas(
                luigi_image, luigi_x, luigi_y, base_image_size[0], base_image_size[1]
            )
            canvas_with_luigi.paste(cropped_luigi, (cropped_x, cropped_y), cropped_luigi)
            luigi_bbox = get_visible_bounding_box(cropped_luigi, cropped_x, cropped_y)
            update_visibility_mask(visibility_mask, luigi_mask, cropped_x, cropped_y)

            if luigi_bbox:
                writer.writerow([f"sample_{sample_idx}_with_luigi", "luigi", *luigi_bbox])

            # Place other characters
            num_characters = random.randint(min_other_characters, max_other_characters)
            for char_idx in range(num_characters):
                char_name, char_img = random.choice(list(characters.items()))
                char_mask = np.array(char_img)[:, :, 3] > 0  # Non-transparent pixels
                char_width, char_height = char_img.size
                x = random.randint(-char_width, base_image_size[0])
                y = random.randint(-char_height, base_image_size[1])
                cropped_img, (cropped_x, cropped_y) = crop_to_canvas(
                    char_img, x, y, base_image_size[0], base_image_size[1]
                )
                canvas_with_luigi.paste(cropped_img, (cropped_x, cropped_y), cropped_img)
                canvas_without_luigi.paste(cropped_img, (cropped_x, cropped_y), cropped_img)
                bbox = (cropped_x, cropped_y, cropped_x + cropped_img.width, cropped_y + cropped_img.height)
                writer.writerow([f"sample_{sample_idx}_without_luigi", char_name, bbox[0], bbox[1], bbox[2], bbox[3]])
                update_visibility_mask(visibility_mask, char_mask, cropped_x, cropped_y)

            # Draw bounding box around Luigi as the last step
            if draw_bounding_box and luigi_bbox:
                draw = ImageDraw.Draw(canvas_with_luigi)
                draw.rectangle(luigi_bbox, outline="red", width=3)

            canvas_with_luigi.save(os.path.join(luigi_folder, f"sample_{sample_idx}_with_luigi.png"), "PNG")
            canvas_without_luigi.save(os.path.join(background_folder, f"sample_{sample_idx}_without_luigi.png"), "PNG")