# 🌿 LeafGuard Model

LeafGuard is an AI-powered apple plant classification system designed to identify plant diseases from leaf images. Utilizing deep learning, the model provides accurate classifications and suggested remedies to help farmers and plant enthusiasts maintain plant health. The model supports three apple plant diseases that is Apple Scab, Apple Cedar Rust and Apple Black Rot.

---

## 🚀 Key Features
- 📷 **Image Classification**: Identifies plant diseases by analyzing leaf images, providing real-time insights into plant health.
- 🤖 **AI-Driven Symptom Analysis**: Uses artificial intelligence to recognize symptoms and suggest possible disease prevention methods.
- 📊 **Data-Driven Predictions**: Trained on a large dataset of plant images to accurately identify diseases such as blight, rust, and mildew.
- 🎯 **Model Training & Evaluation**: The CNN model, based on **ResNet50**, is trained and evaluated using metrics like accuracy, precision, and recall.
- 🖥 **FastAPI-Based Backend**: A robust API service for model inference and integration with applications.

---

## 🛠️ Technologies Used
- **TensorFlow/Keras**: For deep learning model development and training.
- **FastAPI**: For deploying the model as a high-performance web service.
- **Pandas/NumPy**: For data preprocessing and handling.
- **Gemini**: For identifying the plant symptoms and providing preventive measures.


---

## ⚙️ How It Works
1. **Data Collection**: A curated dataset containing images of diseased and healthy leaves is used for training.
2. **Model Training**: A **ResNet50-based** CNN is trained to distinguish between different plant diseases.
3. **API Deployment**: The trained model is served using **FastAPI**, allowing applications to send images and receive disease predictions.
4. **Gemini Response Format**: The API returns JSON responses with classification results, confidence scores, and recommended actions.

---

## 🔧 API Setup & Usage
### Prerequisites
- Python 3.8+
- TensorFlow/Keras installed
- FastAPI and dependencies installed
- Google Gemini

### Clone the Repository
```sh
git clone https://github.com/Dbriane208/LeafGuard-Model.git
cd LeafGuard-Model
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the API Server
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Sample API Request
Send a **POST** request to `/predict` with an image file:
```sh
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@leaf.jpg'
```

### Example API Response

This is the actual sample of the Postman API Response. The FastAPI endpoint is hosted in [Render.com](https://render.com/)

<div style="display:flex;">
    <img src="https://github.com/Dbriane208/LeafGuard/blob/main/leafguard/assets/screenshots/api.png" alt="api" />
</div>

```json
{
  "predicted_class": "Leaf Blight",
  "confidence": 0.9999,
  "symptoms": "Brown spots with yellow halos",
  "measures": "Use fungicides and remove affected leaves"
}
```

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 🤝 Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## 📩 Contact
For inquiries, reach out via email:
📧 **db9755949@gmail.com**

Or connect on LinkedIn: [Daniel Brian Gatuhu](https://www.linkedin.com/in/danielbriangatuhu/)

---

⭐ If you like this project, don't forget to give it a star on GitHub! ⭐

