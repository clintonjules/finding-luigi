from generate_dataset import *
import os
import yaml

def data():
    print("Generating Dataset...")
    
    print("Creating images and annotations...")
    
    coco_annotations_dict = generate_object_detection_dataset(
        num_examples_to_generate=10000,
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


def train():
    print("-" * 100)
    print("Training model...")
    
    # yolov5l.pt
    os.system("python src/yolov5/train.py --data src/data.yaml --epochs 50 --weights '' --cfg yolov5m.yaml --batch-size -1 --project model --name finding_luigi")
    
    print("Model trained")
    print("-" * 100)


def find_luigi(video):
    print("-" * 100)
    print("Searching for Luigi...")
    
    os.system(f"python src/yolov5/detect.py --weights model/finding_luigi/weights/best.pt --source {video} --conf-thres 0.7 --iou-thres 0.5 --save-txt --save-conf --project found_ouput --name luigi_detection --classes 1")
    
    print("Search completed")
    print("-" * 100)
    
    # python src/yolov5/detect.py \
    # --weights custom_output_dir/my_experiment/weights/best.pt \  # Path to your trained weights
    # --source path/to/video.mp4 \                               # Path to the input video
    # --conf-thres 0.4 \                                         # Confidence threshold for detections
    # --iou-thres 0.45 \                                         # IOU threshold for NMS
    # --save-txt \                                               # Save detections as .txt files
    # --save-conf \                                              # Save confidence scores
    # --project inference_output_dir \                           # Directory to save results
    # --name luigi_detection \                                   # Name of the inference folder
    # --classes 1                                                # (Optional) Detect Luigi's class index

def main():
    data()
    
    train()
    
    
if __name__ == "__main__":
    main()
