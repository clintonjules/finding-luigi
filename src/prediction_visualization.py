import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision


# Load the trained model
def get_trained_model(model_path, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set model to evaluation mode
    return model


# Perform prediction and visualize the bounding box with class labels and confidence scores
def predict_and_visualize(image_path, model, device, threshold=0.5):
    # Class label mapping (adjust as per your dataset)
    class_names = {0: "not luigi", 1: "luigi"}  # Add other classes if necessary

    # Load and preprocess the image
    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform prediction
    model.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract predictions above the confidence threshold
    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()
    selected_indices = scores >= threshold

    selected_boxes = boxes[selected_indices]
    selected_scores = scores[selected_indices]
    selected_labels = labels[selected_indices]

    # Visualize the image and bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, score, label in zip(selected_boxes, selected_scores, selected_labels):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add class label and confidence score as text
        class_label = class_names.get(label, "Unknown")
        text_label = f"{class_label}: {score:.2f}"
        ax.text(
            x_min,
            y_min - 10,
            text_label,
            color="red",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    plt.title("Predicted Bounding Boxes with Class Labels and Confidence Scores")
    plt.axis("off")
    plt.show()


# Main execution
def visualize():
    # Parameters
    num_classes = 2  # 'luigi' + background
    model_path = "fasterrcnn_luigi.pth"  # Path to the saved model
    image_path = "data/images/luigi_sample_0.png"  # Path to the image to predict
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the trained model
    model = get_trained_model(model_path, num_classes)

    # Perform prediction and visualize
    predict_and_visualize(image_path, model, device, threshold=0.5)


visualize()
