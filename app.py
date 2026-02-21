import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Ensure src/ is in the python path to import model.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from model import get_dog_classifier
except ImportError:
    st.error("Error: Could not import model logic. Ensure 'src/model.py' exists.")
    st.stop()

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")

@st.cache_resource
def load_model():
    model = get_dog_classifier(num_classes=120, pretrained=False)
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        # We load CPU if CUDA isn't strictly available to avoid issues
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def get_class_names():
    # If the model trained with ImageFolder, the classes are the folder names.
    # We can read them from the data/raw/train directory or hardcode them
    train_dir = 'data/raw/train'
    if os.path.exists(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        # Clean up the Stanford Dog class names (e.g., 'n02085620-Chihuahua' -> 'Chihuahua')
        clean_classes = [c.split('-')[-1].replace('_', ' ').title() for c in classes]
        return classes, clean_classes
    return [], []

def predict(image, model, classes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    return top3_prob, top3_catid

# App UI
st.title("🐾 Dog Breed Classifier")
st.markdown("Upload a photo of a dog and the MobileNetV2 model will predict its breed!")

model = load_model()
raw_classes, clean_classes = get_class_names()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("")
    st.markdown("### Analyzing image...")
    
    if len(clean_classes) == 120 and os.path.exists('models/best_model.pth'):
        # Predict
        top3_prob, top3_catid = predict(image, model, raw_classes)
        
        # Display Results
        st.success(f"Prediction: **{clean_classes[top3_catid[0].item()]}** ({top3_prob[0].item()*100:.1f}%)")
        
        st.markdown("#### Top 3 Predictions:")
        for i in range(3):
            breed = clean_classes[top3_catid[i].item()]
            prob = top3_prob[i].item() * 100
            st.write(f"{i+1}. {breed} - {prob:.2f}%")
            st.progress(int(prob))
    else:
        st.error("Model or classes not found! Please train the model first by running `src/train.py`")
        
st.markdown("---")
st.markdown("<small>Powered by PyTorch, Streamlit, and MobileNetV2.</small>", unsafe_allow_html=True)
