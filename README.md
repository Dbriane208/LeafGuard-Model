# LeafGuard
LeafGuard, a plant Classification model, is a deep learning model designed for plant disease classification. The model is trained to identify various plant diseases based on images of leaves. It utilizes a Convolutional Neural Network (CNN) architecture to process and classify plant images into predefined categories.

# Key Features:
- **Image Classification**: Classifies plant diseases by analyzing leaf images, providing real-time insights into the health of plants.
- **AI-driven Symptom Analysis**: Integrates artificial intelligence to identify symptoms and suggest possible disease prevention methods.
- **Data-Driven Predictions**: Uses a large dataset of plant images to accurately identify diseases such as blight, rust, and mildew.
- **Training and Evaluation**: The model is trained using well-known datasets and evaluated on various metrics such as accuracy, precision, and recall.

# Technologies Used:
- **TensorFlow/Keras**: For model development and training.
- **Fast API**: For deploying the model as a web service (if applicable).
- **Pandas/NumPy**: For data preprocessing and handling.

# How It Works:
- **Data Collection**: A dataset containing images of leaves affected by various diseases is used for training.
- **Model Training**: The CNN model is trained to learn the distinguishing features of each disease used Resnet50.
- **Deploymen**t: The trained model can be integrated into applications to detect diseases from new leaf images.

# Future Enhancements:
- Integration with IoT sensors to detect environmental factors influencing plant health.
- Real-time disease prediction via a mobile application.
- Expansion to other types of plants and diseases.
