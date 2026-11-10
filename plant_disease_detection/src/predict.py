
import os
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from model import PlantDiseaseCNN


# 1. DEVICE CONFIGURATION

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# 2. MODEL PATH CONFIGURATION

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model.pth"
)



# 3. CLASS NAMES

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]



# 4. LOAD TRAINED MODEL

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

# Load model weights
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

print("\nCheckpoint loaded successfully!")


# 5. DETECT NUMBER OF CLASSES

num_classes = state_dict["classifier.4.weight"].shape[0]

print("Detected classes:", num_classes)

# Check class names
if len(class_names) != num_classes:
    raise ValueError(
        f"Number of class names ({len(class_names)}) "
        f"does not match model output classes ({num_classes}). "
        "Please correct class_names."
    )


# 6. INITIALIZE MODEL

model = PlantDiseaseCNN(
    num_classes=num_classes
)

model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("\nModel loaded successfully!")
print("Number of classes:", num_classes)


# 7. IMAGE PREPROCESSING

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# 8. PREDICTION FUNCTION

def predict_image(image_path):

    # Check image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move image to device
    image_tensor = image_tensor.to(device)

    # Model prediction
    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    # Get predicted class
    predicted_index = predicted.item()

    predicted_class = class_names[predicted_index]

    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score


# 9. TEST IMAGE PREDICTION

if __name__ == "__main__":

    image_path = input(
        "\nEnter image path: "
    ).strip().strip('"')

    try:

        predicted_class, confidence = predict_image(
            image_path
        )

        print("     PLANT DISEASE PREDICTION")

        print(
            "Predicted Disease:",
            predicted_class
        )

        print(
            f"Confidence Score: {confidence:.2f}%"
        )

    except Exception as e:

        print("Prediction Error:", e)
