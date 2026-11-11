
import os
import io
import socket
import ipaddress
import requests
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from urllib.parse import urlparse
from torchvision import transforms

from model import PlantDiseaseCNN


# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# 2. CONFIGURATION

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "model.pth"
    )
)

IMAGE_SIZE = 224
NUM_CLASSES = 15

print("Using device:", DEVICE)


# 3. CLASS NAMES

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


# 4. IMAGE TRANSFORMATION

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5. LOAD TRAINED MODEL

@st.cache_resource
def load_model():

    model = PlantDiseaseCNN(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")

    return model


# 6. DOWNLOAD IMAGE FROM URL

def download_image(url):

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            "Only HTTP and HTTPS URLs are allowed."
        )

    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid image URL.")

    try:

        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (
                443 if parsed.scheme == "https" else 80
            )
        )

        for address in addresses:

            ip = ipaddress.ip_address(
                address[4][0]
            )

            if not ip.is_global:
                raise ValueError(
                    "Private or local image URLs are not allowed."
                )

    except socket.gaierror:

        raise ValueError(
            "Could not resolve image URL."
        )

    response = requests.get(
        url,
        timeout=15,
        stream=True,
        allow_redirects=False,
        headers={
            "User-Agent": "PlantDiseaseDetector/1.0"
        }
    )

    if response.status_code != 200:
        raise ValueError(
            "Unable to download image. Check the URL."
        )

    content_type = response.headers.get(
        "Content-Type", ""
    )

    if not content_type.lower().startswith("image/"):
        raise ValueError(
            "The URL does not point to an image."
        )

    content = bytearray()

    for chunk in response.iter_content(8192):

        content.extend(chunk)

        if len(content) > 10 * 1024 * 1024:
            raise ValueError(
                "Image must be smaller than 10 MB."
            )

    return Image.open(
        io.BytesIO(content)
    ).convert("RGB")


# 7. PREDICT DISEASE

def predict_disease(image, model):

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()

    disease_name = CLASS_NAMES[predicted_index]

    confidence_value = confidence.item() * 100

    if "healthy" in disease_name.lower():

        status = "HEALTHY"

    else:

        status = "DISEASE DETECTED"

    top_values, top_indices = torch.topk(
        probabilities[0],
        k=3
    )

    top_predictions = []

    for value, index in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):

        name = CLASS_NAMES[index].replace(
            "___", " - "
        ).replace(
            "__", " "
        ).replace(
            "_", " "
        )

        top_predictions.append(
            (name, value * 100)
        )

    return (
        disease_name,
        status,
        confidence_value,
        top_predictions
    )


# 8. CUSTOM CSS

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        color: #16834A;
        font-size: 38px;
        font-weight: bold;
    }

    .sub-title {
        text-align: center;
        color: #777777;
        font-size: 17px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# 9. APPLICATION HEADER

st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-powered Plant Leaf Health Analysis using CNN</div>',
    unsafe_allow_html=True
)

st.divider()


# 10. LOAD MODEL

try:

    model = load_model()

except Exception as e:

    st.error(
        f"Unable to load model: {str(e)}"
    )

    st.stop()


# 11. IMAGE INPUT

st.subheader("📸 Upload Plant Leaf Image")

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

with col2:

    image_url = st.text_input(
        "Or enter image URL",
        placeholder="https://example.com/leaf.jpg"
    )


# 12. GET AND DISPLAY IMAGE

image = None

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error("Invalid uploaded image.")

elif image_url.strip():

    try:

        with st.spinner("Downloading image..."):

            image = download_image(
                image_url
            )

    except Exception as e:

        st.error(str(e))


# Display image in a smaller centered layout

if image is not None:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.image(
            image,
            caption="Selected Plant Leaf Image",
            width=300
        )


# 13. PREDICTION BUTTON

if st.button(
    "🔍 Predict Disease",
    type="primary",
    use_container_width=True
):

    if image is None:

        st.warning(
            "Please upload an image or provide a valid image URL."
        )

    else:

        with st.spinner(
            "Analyzing plant leaf..."
        ):

            try:

                (
                    disease_name,
                    status,
                    confidence,
                    top_predictions
                ) = predict_disease(
                    image,
                    model
                )

                formatted_name = (
                    disease_name
                    .replace("___", " - ")
                    .replace("__", " ")
                    .replace("_", " ")
                )

                st.divider()

                st.subheader(
                    "🌱 Prediction Results"
                )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    if status == "HEALTHY":

                        st.success(
                            "🌿 Healthy Leaf"
                        )

                    else:

                        st.error(
                            "⚠️ Disease Detected"
                        )

                    st.markdown(
                        f"### {formatted_name}"
                    )

                with result_col2:

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.progress(
                        confidence / 100
                    )

                st.divider()

                st.subheader(
                    "📊 Top 3 Predictions"
                )

                for name, score in top_predictions:

                    st.write(
                        f"**{name}** — {score:.2f}%"
                    )

                    st.progress(
                        score / 100
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {str(e)}"
                )


# 14. FOOTER

st.divider()

st.caption(
    "Plant Disease Detection | PyTorch CNN | Streamlit"
)

st.caption(
    "Verify results before making agricultural treatment decisions."
)
