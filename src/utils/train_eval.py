import os
import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, device,
                checkpoint_dir="./checkpoints", save_best=True):
    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

        # Save best model
        if save_best and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Best model saved (Val Acc: {best_acc:.2f}%)")

def evaluate_model(model, loader, criterion, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total, loss_sum / len(loader)
