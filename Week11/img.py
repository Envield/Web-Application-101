import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)  # Adjust the output layer to match your number of classes
model.load_state_dict(torch.load("fruit_classification_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define image transformation
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names (ensure they match the ones used during training)
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']  # Update with actual class names

# Function to predict image class and confidence
def predict_image(image):
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), top_confidence.item() * 100  # Return confidence as percentage

# Streamlit UI
st.markdown(
        """
        <style>
        /* Ubah warna background aplikasi */
        .stApp {
            background-color: #17222B;
        }

        /* Tambahkan padding untuk halaman */
        .block-container {
            padding-top: 4rem;
            padding-bottom: 6rem;
            padding-left: 16rem;
            padding-right: 16rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


col1, col2 = st.columns([3, 4])  

with col1:

        st.markdown("")
        st.title(":green[Fruit Classification] Dashboard🍉🥭")
        st.markdown("> <span style='color:white'>Simply upload a fruit image, and within seconds, you'll know whether it's an Apple, Banana, Grape, Mango, or Strawberry</span>", unsafe_allow_html=True)
        st.markdown("<span>**Welcome!** This dashboard allows you to upload images of fruits, and our model will classify them with high accuracy and provide confidence scores.</span>", unsafe_allow_html=True)


with col2:

        st.image("./gif/Fruit.gif", use_column_width=True)

st.divider()


col1, col2 = st.columns([2, 3])  

with col1:

        st.image("./gif/week11 (3).gif", use_column_width=True)


with col2:

        st.subheader("Before you upload...")
        st.markdown("> <span style='color:white'>The images that can be used for classification should contain one or more of the following fruits: </span>", unsafe_allow_html=True)
        st.markdown("""
                    1. Apple
                    2. Banana
                    3. Grape
                    3. Mango
                    5. Strawberry
                    """
                    )
        st.success('You may still upload other images, but the results might not match your expectations.')
        st.markdown("<span style='color:white'>We recommend preparing your images in one folder and naming each file appropriately for easier uploading below⬇️</span>", unsafe_allow_html=True)
        st.markdown("- Lastly, we hope you enjoy using our service, and thank you for choosing us!🍏🍇🍌")

st.divider()
# Upload multiple images
uploaded_files = st.file_uploader("Choose images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Capture image from camera
camera_image = st.camera_input("You can also Take a picture of your Fruits!🍏🍇🍌")

# List to store images for prediction
images_to_predict = []

# Add uploaded images to list
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        images_to_predict.append(image)

# Add camera image to list
if camera_image:
    image = Image.open(camera_image).convert('RGB')
    images_to_predict.append(image)

# Display predictions if any images are present  
if uploaded_files:
    st.divider()
    st.write("### Predictions Results")
    cols = st.columns(3)  # Adjust for layout

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert('RGB')
        predicted_class, confidence = predict_image(image)

        # Display image with prediction
        with cols[idx % 3]:  # Display in columns
            st.image(image, caption=f"{class_names[predicted_class]} ({confidence:.2f}%)", use_column_width=True)
            st.write(f"Predicted: {class_names[predicted_class]}")
            st.write(f"Confidence: {confidence:.2f}%")
