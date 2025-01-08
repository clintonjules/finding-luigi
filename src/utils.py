import numpy as np
from PIL import Image
import random
   

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

def random_overlay(base_image_path: str, overlay_image_path: str, output_image_path: str = None, return_overlay_values: bool = False) -> tuple[Image.Image, int, int]:
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
    if output_image_path:
        combined_image.save(output_image_path, format="PNG")

    if return_overlay_values:
        return combined_image, x_offset, y_offset, overlay_image_area, base_image_area, overlapping_area
    
    return combined_image


def random_overlay_multiple(base_image_path: str, overlay_image_paths: list, output_image_path: str = None, 
                            max_overlap_percentage: float = 0.5, max_retries: int = 10, 
                            max_base_coverage: float = 0.8, return_overlay_values: bool = False) -> Image.Image:
    """
    Overlay multiple images randomly on top of a base image while limiting overlap percentage and base coverage.

    Args:
        base_image_path (str): Path to the base image.
        overlay_image_paths (list): List of paths to overlay images.
        output_image_path (str): Path to save the output image (optional).
        max_overlap_percentage (float): Maximum allowed overlap percentage for each overlay.
        max_retries (int): Maximum number of retries to fit an overlay within the threshold.
        max_base_coverage (float): Maximum allowed base coverage percentage.
        return_overlay_values (bool): Whether to return detailed overlay values.

    Returns:
        Image.Image: The combined image with all overlays applied.
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

    while retries < max_retries:
        retries += 1

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

                # Check if the overlap is within the allowed threshold
                if overlap_percentage <= max_overlap_percentage:
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

            if not overlay_placed:
                print(f"Warning: Failed to place overlay {overlay_image_path} within {max_retries} retries.")

        # Calculate the total base coverage percentage
        valid_base_pixels_covered = base_pixels_covered.intersection(valid_base_pixels)
        base_coverage_percentage = len(valid_base_pixels_covered) / base_image_area

        # Break if base coverage is within the allowed limit
        if base_coverage_percentage <= max_base_coverage:
            break

    if base_coverage_percentage > max_base_coverage:
        print(f"Warning: Failed to keep base coverage within {max_base_coverage * 100}% after {retries} retries.")

    # Save the resulting image
    if output_image_path:
        combined_image.save(output_image_path, format="PNG")

    if return_overlay_values:
        return combined_image, {base_image_path: {"base_image_area": base_image_area, "base_coverage_percentage": base_coverage_percentage}}, overlapping_values

    return combined_image