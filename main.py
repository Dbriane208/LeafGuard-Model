from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
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
import base64
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

# Configure the Gemini API (for symptoms and measures only)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Loading the class names
CLASS_NAMES = [ 
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy"
]

# Confidence threshold - if model confidence is below this, image is likely not a potato leaf
CONFIDENCE_THRESHOLD = 0.60

def is_potato_leaf(image_data: bytes) -> tuple[bool, str]:
    """
    Use Gemini vision API to verify if the image contains a potato leaf.
    Returns (is_valid, message).
    """
    try:
        # Convert image bytes to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine MIME type
        image = Image.open(BytesIO(image_data))
        mime_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
        
        prompt = """Analyze this image and determine if it shows a potato leaf (or potato plant leaves).
        
        Respond with ONLY one of these exact responses:
        - "YES" if the image clearly shows a potato leaf or potato plant leaves (healthy or diseased)
        - "NO" if the image does not show a potato leaf
        
        Do not include any other text or explanation."""
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ]
        )
        
        result = response.text.strip().upper()
        
        if "YES" in result:
            return True, "Valid potato leaf image"
        else:
            return False, "The image does not appear to contain a potato leaf. Please upload a clear image of a potato leaf."
            
    except Exception as e:
        print(f"Gemini image validation error: {e}")
        # If Gemini fails, allow the image to proceed to the model
        return True, "Validation skipped due to API error"

# Fallback symptoms and measures when Gemini API fails
FALLBACK_INFO = {
    "Potato Early Blight": {
        "symptoms": "Early blight causes dark brown to black spots with concentric rings (target-like pattern) on older, lower leaves first. The spots may have a yellow halo around them, and severely affected leaves turn yellow and drop prematurely. Stems and tubers can also develop dark, sunken lesions.",
        "measures": "Practice crop rotation with non-solanaceous crops for at least 2-3 years. Remove and destroy infected plant debris and avoid overhead irrigation to keep foliage dry. Apply fungicides preventively when conditions favor disease development (warm, humid weather)."
    },
    "Potato Late Blight": {
        "symptoms": "Late blight appears as water-soaked, pale green to dark brown lesions on leaves that rapidly expand. A white, fuzzy mold growth appears on the underside of leaves during humid conditions. The disease spreads quickly, causing entire plants to collapse within days, and can infect tubers causing firm, brown rot.",
        "measures": "Plant certified disease-free seed potatoes and choose resistant varieties when available. Apply protective fungicides before symptoms appear, especially during cool, wet weather. Remove and destroy infected plants immediately and avoid overhead irrigation."
    },
    "Potato Healthy": {
        "symptoms": "No disease symptoms detected. Your potato plant appears healthy with normal green foliage.",
        "measures": "Continue good agricultural practices: ensure proper spacing for air circulation, water at the base of plants, and monitor regularly for early signs of disease. Maintain balanced soil nutrition and remove any weeds that may harbor pests or diseases."
    }
}

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

def get_symptoms_and_measures(disease_name):
    """
    Get symptoms and prevention measures using Gemini API.
    Falls back to static content if API fails.
    """
    try:
        prompt = f"""
          Describe the symptoms of {disease_name} in a three-sentence paragraph that is short, clear, and concise. 
          Then, provide three effective prevention measures for {disease_name}, each in a separate sentence.
        """

        res = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt]
        )

        # Extracting the response text
        response_text = res.text

        # Splitting symptoms and measures [assuming we have double line break]
        parts = response_text.split("\n\n")  

        symptoms = parts[0].strip() if len(parts) > 0 else FALLBACK_INFO[disease_name]["symptoms"]
        measures = parts[1].strip() if len(parts) > 1 else FALLBACK_INFO[disease_name]["measures"]

        return symptoms, measures
    except Exception as e:
        print(f"Gemini API error: {e}. Using fallback content.")
        fallback = FALLBACK_INFO.get(disease_name, {
            "symptoms": "Unable to retrieve symptoms. Please consult an agricultural expert.",
            "measures": "Unable to retrieve prevention measures. Please consult an agricultural expert."
        })
        return fallback["symptoms"], fallback["measures"]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    
    # Pre-validate: Use Gemini to check if the image is a potato leaf
    is_valid, validation_message = is_potato_leaf(image_bytes)
    if not is_valid:
        return {
            "error": True,
            "response": validation_message
        }

    # Perform model prediction
    image = read_file_as_image(image_bytes)
    prediction = model.predict(image)

    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Check if confidence is too low - likely not a potato leaf
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "error": True,
            "response": "The image does not appear to be a recognizable potato leaf or the image quality is insufficient for classification. Please upload a clear image of a potato leaf."
        }

    # Fetch symptoms and prevention measures (with fallback)
    symptoms, measures = get_symptoms_and_measures(predicted_class)

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "symptoms": symptoms,
        "measures": measures
    }    

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)