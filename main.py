from fastapi import FastAPI, File, UploadFile # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import uvicorn # type: ignore
import numpy as np # type: ignore
from io import BytesIO
from PIL import Image # type: ignore
from keras.models import load_model # type: ignore
from keras.layers import DepthwiseConv2D # type: ignore
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import imghdr


app = FastAPI()

origins = [
    "http://localhost",
    "https://leafguard-model.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the environment variables
load_dotenv()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Using custom load function
def remove_groups_arg(config):
    if 'groups' in config:
        del config['groups']
    return config

# Custom objects
custom_objects = {
    'DepthwiseConv2D': lambda **kwargs: DepthwiseConv2D(**remove_groups_arg(kwargs))
}   

# Load the model
model = load_model(
    "models/keras/keras_model.h5",
    custom_objects=custom_objects,
    compile=False
)

# Loading the class names
CLASS_NAMES = [ 
    "Apple Cedar Rust",
    "Apple Black Rot",
    "Apple Scab",
    "Apple Healthy"
]

@app.get("/")
async def ping():
    return "Hello, Welcome to LeafGuard"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array before normalization
    image = np.array(image)
    
    # Normalize
    image = (image.astype(np.float32) / 127.5) - 1
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def check_supported_image(image):
    # Check if the image is related to supported apple diseases
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = "Analyze the image and determine if it contains any of the following apple diseases: " \
             "Apple Cedar Rust, Apple Black Rot, Apple Scab, or Apple Healthy. " \
             "If it does not match these categories, return 'Error: Image not supported'. " \
             "If it's an apple leaf but has a different disease, return 'Unsupported disease'." \
             "If the leaf is apple related and has our supported disease, return 'Supported disease image statement only'"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image,mime_type="image/jpg")
        ]
    )

    return response.text if response else "Error. Unable to process image"
    

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()

    response = check_supported_image(image_bytes)

    if "Error" in response or "Unsupported disease" in response:
        return {"response": response}

    # Perform model prediction
    image = read_file_as_image(image_bytes)
    prediction = model.predict(image)

    # Get the predicted class
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }    

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)