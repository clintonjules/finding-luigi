import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from PIL import Image
import os
import pandas as pd
import torchvision


# Dataset class
class LuigiDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None, background_images_dir=None):
        """
        Custom Dataset for loading images and bounding boxes.

        Args:
            dataframe: Pandas DataFrame containing bounding box information.
            image_dir: Directory containing images.
            transforms: Optional transformations for data augmentation.
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.background_images = []
        
        # Load background images (no Luigi)
        if background_images_dir:
            self.background_images = [
                os.path.join(background_images_dir, fname)
                for fname in os.listdir(background_images_dir)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

    def __getitem__(self, idx):
        # If idx is for a background image
        if idx >= len(self.dataframe):
            bg_image_path = self.background_images[idx - len(self.dataframe)]
            image = Image.open(bg_image_path).convert("RGB")
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}

            # Apply transforms if specified
            if self.transforms:
                image = self.transforms(image)

            return image, target


        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, os.path.basename(row["image_path"]))
        image = Image.open(image_path).convert("RGB")

        # Bounding box
        boxes = torch.tensor(
            [[row["x_min"], row["y_min"], row["x_max"], row["y_max"]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([1], dtype=torch.int64)  # 1 for 'luigi'
        target = {"boxes": boxes, "labels": labels}

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.dataframe)


# Load pretrained Faster R-CNN model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


# Training and evaluation loops
def train_model(model, dataloader, optimizer, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")


def evaluate_model(model, dataloader, device):
    model.eval()
    print("Evaluating the model...")
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)  # Outputs contain predicted boxes and labels

            # Print evaluation results for one batch as a checkpoint
            print(f"Batch {idx+1} outputs:")
            print(outputs)
            break  # For simplicity, only evaluate one batch


# Main execution
def train(num_classes=2, num_epochs=5, batch_size=4, learning_rate=0.005):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load bounding box CSV file
    bounding_boxes_path = 'data/bounding_boxes.csv'
    bounding_boxes_data = pd.read_csv(bounding_boxes_path)

    # Image directories
    luigi_image_dir = 'data/luigi_images'
    background_image_dir = 'data/background_images'  # Add background images here

    # Transforms
    transforms = T.Compose([T.ToTensor()])

    # Create Dataset and DataLoader
    dataset = LuigiDataset(bounding_boxes_data, luigi_image_dir, transforms, background_image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Load the model
    model = get_model(num_classes).to(device)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    # Train the model
    train_model(model, dataloader, optimizer, num_epochs, device)

    # Save the model
    model_save_path = 'find_luigi_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

train(2, 5, 4, .005)
