from generate_dataset import *
import os
import yaml

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
    
    
    print("Creating yaml file...")


    # Load environment variables
    train_dir = os.getenv("TRAIN_DIR")  # Replace with default path if env not set
    val_dir = os.getenv("VAL_DIR")  # Replace with default path if env not set

    # Define the YAML structure
    data = {
        "train": train_dir,
        "val": val_dir,
        "nc": 4,
        "names": ["mario", "luigi", "wario", "yoshi"]
    }

    # Write to a YAML file
    yaml_path = "src/data.yaml"  # Change the filename/path as needed
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    print(f"YAML file created at {yaml_path}")