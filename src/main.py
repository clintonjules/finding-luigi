from generate_dataset import *
import os

def main():
    print("Generating Dataset...")
    
    print("Creating images and annotations...")
    
    coco_annotations_dict = generate_object_detection_dataset(
        num_examples_to_generate=15,
        num_characters_range=(4, 150),
        save_dir=Path("data").resolve(),
        display_plot=False,
        display_boxes=False,
        proximity_threshold=5,
        pad_inches_possible_values=[-0.75],
    )
    
    print("Images and annotations created")
    
    print("Convert COCO to YOLO...")
    
    os.system("python src/COCO2YOLO.py -j data/annotations.json -o .")
    
    print("COCO converted to YOLO")
    
    print("Splitting dataset...")
    
    split_dataset(
        source_dir="data",  # Replace with your dataset directory
        output_dir="data",  # Replace with the desired output directory
        split_ratio=0.8  # 80% train, 20% validation
    )   
    
    print("Dataset split")
    
    print("Dataset Generated!")
    
    