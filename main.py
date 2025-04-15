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

# Configure the API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
    prompt = """
       Analyze the provided image and determine if it contains any of the following apple leaf conditions: Apple Cedar Rust, Apple Black Rot, Apple Scab, or a Healthy Apple Leaf.

      If the image does not depict an apple leaf, return: 'Error: Image not supported.'
      If the leaf is from an apple tree but has an unsupported disease, return: 'Unsupported disease.'
      If the leaf belongs to an apple tree and has one of the supported diseases,return: 'Supported disease image'

    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image,mime_type="image/jpg")
        ]
    )

    return response.text if response else "Error. Unable to process image"

def get_symptoms_and_measures(disease_name):
    prompt = f"""
      Describe the symptoms of {disease_name} in a three-sentence paragraph that is short, clear, and concise. 
      Then, provide three effective prevention measures for {disease_name}, each in a separate sentence.
    """

    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )

    # Extracting the response text
    response_text = res.text

    # Splitting symptoms and measures [assumming we have double line break]
    parts = response_text.split("\n\n")  

    symptoms = parts[0].strip() if len(parts) > 0 else "Symptoms not found."
    measures = parts[1].strip() if len(parts) > 1 else "Prevention measures not found."

    return symptoms, measures

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

    # Fetch symptoms and prevention measures
    symptoms, measures = get_symptoms_and_measures(predicted_class)

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "symptoms": symptoms,
        "measures": measures
    }    

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)