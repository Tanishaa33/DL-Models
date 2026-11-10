import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader, class_names
from model import PlantDiseaseCNN


# 1. DEVICE CONFIGURATION

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



# 2. MODEL CONFIGURATION

NUM_CLASSES = len(class_names)
EPOCHS = 50
LEARNING_RATE = 0.001

model = PlantDiseaseCNN(num_classes=NUM_CLASSES)
model = model.to(device)


# 3. LOSS FUNCTION

criterion = nn.CrossEntropyLoss()


# 4. OPTIMIZER

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 5. SAVE MODEL DIRECTORY

os.makedirs("models", exist_ok=True)

BEST_MODEL_PATH = "models/plant_disease_model.pth"


# 6. TRAINING FUNCTION

def train_one_epoch(model, train_loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        running_loss += loss.item()

        # Calculate predictions
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# 7. VALIDATION FUNCTION

def validate_model(model, val_loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Calculate predictions
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)

    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy


# 8. TRAINING HISTORY

best_val_accuracy = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "epoch_time": []
}


# 9. START TOTAL TRAINING TIMER

total_start = time.time()


# 10. MAIN TRAINING LOOP

for epoch in range(EPOCHS):

    epoch_start = time.time()

    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    
    
    

    # Training

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    # Validation

    val_loss, val_accuracy = validate_model(
        model,
        val_loader,
        criterion
    )



    # Calculate Epoch Time

    if device.type == "cuda":
        torch.cuda.synchronize()

    epoch_time = time.time() - epoch_start


    # Store Training History

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    history["epoch_time"].append(epoch_time)


    # Display Results


    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Epoch Training Time: {epoch_time / 60:.2f} minutes")

    
    # Save Best Model

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({

            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": NUM_CLASSES,
            "epoch": epoch + 1,
            "val_accuracy": val_accuracy

        }, model.pth)

        print("Best model saved successfully!")


# 11. TOTAL TRAINING TIME

if device.type == "cuda":
    torch.cuda.synchronize()

total_time = time.time() - total_start

total_minutes = total_time / 60
total_hours = total_time / 3600


# 12. FINAL TRAINING RESULTS

print("TRAINING COMPLETED SUCCESSFULLY!")


print(f"Total Epochs: {EPOCHS}")

print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

print(f"Total Training Time: {total_minutes:.2f} minutes")

print(f"Total Training Time: {total_hours:.2f} hours")

print(f"Average Time Per Epoch: {sum(history['epoch_time']) / EPOCHS / 60:.2f} minutes")

print("Model saved at:", model.pth)
