import streamlit as st
import numpy as np
import cv2
from cv2 import dnn
from PIL import Image
import tempfile

# Model files
proto_file = "Model/colorization_deploy_v2.prototxt"
model_file = "Model/colorization_release_v2.caffemodel"
hull_pts = "Model/pts_in_hull.npy"

# Load the pre-trained model
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

# Set up the model layers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image):
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize image for the network
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # Mean subtraction

    # Predict the ab channels from the input L channel
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    # Take the L channel and combine with predicted ab channel
    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Convert to 0-255 range and uint8
    colorized = (255 * colorized).astype("uint8")
    colorized_img = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    return colorized_img

# Streamlit app
st.title("Image Colorizer App")
st.write("Upload a grayscale image to see it colorized!")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform colorization
    with st.spinner("Colorizing... Please wait..."):
        colorized_image = colorize_image(image)

    # Display the colorized image
    st.image(colorized_image, caption="Colorized Image", use_column_width=True)

    # Save colorized image to download
    colorized_pil = Image.fromarray(cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    colorized_pil.save(temp_file.name)

    st.download_button(
        label="Download Colorized Image",
        data=open(temp_file.name, "rb").read(),
        file_name="colorized_image.png",
        mime="image/png"
    )
