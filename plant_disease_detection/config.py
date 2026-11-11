
import os
import torch

# PROJECT CONFIGURATION
PROJECT_NAME = "Plant Disease Detection"

# DIRECTORY PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "PlantVillage"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.pth"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

# MODEL CONFIGURATION

IMAGE_SIZE = 224
NUM_CLASSES = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50


# DATASET CONFIGURATION
TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42



# CLASS NAMES
CLASS_NAMES = [
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
    "Tomato_healthy"
]

