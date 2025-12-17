import imghdr
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

# Feature flags
USE_GEMINI_VALIDATION = os.getenv("USE_GEMINI_VALIDATION", "false").lower() == "true"
USE_GEMINI_DETAILS = os.getenv("USE_GEMINI_DETAILS", "false").lower() == "true"

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

# Configure the API with fallback
def get_client():
    """Get a Gemini client with fallback to secondary API key if primary fails"""
    if not USE_GEMINI_VALIDATION and not USE_GEMINI_DETAILS:
        return None
        
    primary_key = os.getenv("GEMINI_API_KEY")
    secondary_key = os.getenv("GEMINI_API_KEY_2")
    
    try:
        client = genai.Client(api_key=primary_key)
        return client
    except Exception as e:
        if secondary_key:
            try:
                client = genai.Client(api_key=secondary_key)
                return client
            except Exception as e2:
                return None
        return None

client = get_client()

# Loading the class names
CLASS_NAMES = [ 
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healty"
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
    if not USE_GEMINI_VALIDATION or client is None:
        return "Supported disease image"  # Skip validation when Gemini is disabled
        
    try:
        prompt = """
           Analyze the provided image and determine if it contains any of the following potato leaf conditions: Potato Early Blight, Potato Late Blight, or a Potato Healthy Leaf.

          If the image does not depict a potato leaf, return: 'Error: Image not supported.'
          If the leaf is from a potato plant but has an unsupported disease, return: 'Unsupported disease.'
          If the leaf belongs to a potato plant and has one of the supported diseases,return: 'Supported disease image'
          If the leaf is potato healthy then for symptoms return: 'No symptoms identified. Your plant is healthy.'
          If the leaf is potato healthy then for measures return: 'No measures given. Your plant is doing right.'

        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=image,mime_type="image/jpg")
            ]
        )

        return response.text if response else "Error. Unable to process image"
    except Exception as e:
        print(f"Gemini API error in validation: {e}")
        return "Supported disease image"  # Continue with prediction on error

def get_symptoms_and_measures(disease_name):
    # Fallback static content when Gemini is unavailable
    static_info = {
        "Potato Early Blight": {
            "symptoms": "Early blight causes dark brown spots with concentric rings on older leaves. The spots gradually enlarge and may cause leaves to yellow and drop prematurely. Severe infections can reduce yield and affect tuber quality.",
            "measures": "Rotate crops with non-host plants for at least 2-3 years. Apply fungicides preventively, especially during warm, humid weather. Remove and destroy infected plant debris to reduce disease spread."
        },
        "Potato Late Blight": {
            "symptoms": "Late blight appears as water-soaked lesions on leaves that quickly turn brown or black. White fuzzy growth may appear on leaf undersides during humid conditions. The disease can rapidly destroy entire plants and spread to tubers.",
            "measures": "Plant certified disease-free seed potatoes and resistant varieties when possible. Apply protective fungicides before symptoms appear, especially in wet conditions. Destroy infected plants immediately and avoid overhead irrigation."
        },
        "Potato Healty": {
            "symptoms": "No symptoms identified. Your plant is healthy.",
            "measures": "No measures given. Your plant is doing right."
        }
    }
    
    if not USE_GEMINI_DETAILS or client is None:
        info = static_info.get(disease_name, {
            "symptoms": "Information not available.",
            "measures": "Please consult agricultural resources for specific guidance."
        })
        return info["symptoms"], info["measures"]
    
    try:
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
    except Exception as e:
        print(f"Gemini API error in details: {e}")
        info = static_info.get(disease_name, {
            "symptoms": "Information not available.",
            "measures": "Please consult agricultural resources for specific guidance."
        })
        return info["symptoms"], info["measures"]

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