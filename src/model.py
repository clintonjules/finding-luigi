from fastai.vision.all import *
from pathlib import Path


def train_model(
    dataset_path: str = 'data',
    save_model_path: str = 'find_luigi_model',
    validation_pct: float = 0.2,
    epochs: int = 50,
    batch_size: int = 64,
    base_model: callable = resnet50,
    item_tfms: callable = Resize(224)
):
    """
    Train a CNN model using a dataset.

    Args:
        dataset_path (str): Path to the dataset folder (organized by class folders).
        save_model_path (str): Path to save the trained model.
        validation_pct (float): Percentage of data to use for validation.
        epochs (int): Number of fine-tuning epochs.
        batch_size (int): Batch size for the DataLoaders.
        base_model (callable): CNN model architecture to use (e.g., resnet34).
        item_tfms (callable): Transformations to apply to items (default: Resize(224)).

    Returns:
        None
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist!")

    # Load the data
    print("Loading dataset...")
    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=validation_pct,
        item_tfms=item_tfms,
        bs=batch_size
    )

    # Create a CNN learner
    print("Creating CNN learner...")
    model = vision_learner(dls, base_model, metrics=[accuracy, error_rate])

    # Find the best learning rate
    print("Finding best learning rate...")
    lr_valley = model.lr_find().valley

    # Fine-tune the model
    print(f"Training the model for {epochs} epochs...")
    model.fine_tune(epochs, base_lr=lr_valley, freeze_epochs=5)

    # Save the model for training continuation
    print(f"Saving model weights to {save_model_path}.pth...")
    model.save(save_model_path)

    # Export the model for inference
    print(f"Exporting model to {save_model_path}.pkl for deployment...")
    model.export(f"{save_model_path}.pkl")

    print("Training and saving completed successfully.")