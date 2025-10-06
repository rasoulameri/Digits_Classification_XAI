import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.cnn_model import CNNModel
from src.utils.dataset import get_dataloaders
from src.utils.train_eval import train_model, evaluate_model
from src.utils.explain import gradcam

def load_config(path="./config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=config["data"]["train_dir"],
        test_dir=config["data"]["test_dir"],
        batch_size=config["data"]["batch_size"]
    )

    model = CNNModel(num_classes=config["model"]["num_classes"])
    criterion = nn.CrossEntropyLoss()

    if config["train"]["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )

    os.makedirs(config["train"]["checkpoint_dir"], exist_ok=True)
    # best_path = os.path.join(config["train"]["checkpoint_dir"], "best_model.pth")

    train_model(
        model,
        train_loader,
        val_loader,
        config["train"]["epochs"],
        optimizer,
        criterion,
        device,
        checkpoint_dir=config["train"]["checkpoint_dir"],
        save_best=config["train"]["save_best"]
    )

    test_acc, _ = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
