# 🧠 NeuroSpace: AI for Astronaut Mental Health Monitoring

NeuroSpace is a smart, non-intrusive system powered by **7 AI models** to monitor the physical and psychological state of astronauts during space missions.

### 🌌 Features:
- Vital Signs Monitoring (DNN, PyTorch)
- Sleep & Fatigue Analysis (XGBoost)
- Bone Density Risk (XGBoost + Neural Network)
- Speech Emotion Detection (Keras)
- Facial Expression Recognition (CNN, FER-2013)
- Text-based Psychological Analysis (TF-IDF + Logistic Regression)
- Final GPT-based summary report (Fine-tuned GPT-3.5)

### 📊 Output:
The system automatically generates a personalized, readable health scan report and sends alerts to the control center if risk is detected.

---

Made for Hackathon: **ابتكارثون | ابتكار في الفضاء**

🚀 Designed by: NeuroSpace Team
--------------------------------
## 🔧 How to Use the Trained Models

Each folder under `trained_models/` contains the necessary files for one of the seven AI models used in our **NeuroSpace** project.

### 📁 Folder Structure

Each folder is named based on the model number and its task:

```
trained_models/
├── Model_1_Vital_Signs/
├── Model_2_Sleep_Pattern/
├── Model_3_Bone_Health/
├── Model_4_Speech_Emotion/
├── Model_5_Facial_Expression/
├── Model_6_Text_Analysis/
├── Model_7_AI_Report_Generation/
```

---

### 🧠 How to Load the Model Files

#### `.pkl` files  
Used for encoders, scalers, and some ML models. Load using:

```python
import pickle

with open("filename.pkl", "rb") as f:
    obj = pickle.load(f)
```

#### `.h5` files (Keras Models)

```python
from tensorflow.keras.models import load_model

model = load_model("model_name.h5")
```

#### `.json + .h5` combo (Model 5 – Facial Expression)

```python
from tensorflow.keras.models import model_from_json

# Load model structure
with open("facialemotionmodel.json", "r") as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")
```

> 📝 **Note**: Model 5 `.ipynb` file is not available due to data loss, but you can load and use the model with the files above.

---

### 🧪 How to Run the Final Integrated Model

To run the final system that combines all 6 models and generates the AI report (Model 7):

1. Go to the `notebooks/` folder and open:

```
all_the_models_combined.ipynb
```

2. Make sure the `trained_models/` folder is in the same directory.
3. Install required libraries (if not already installed):

```bash
pip install numpy pandas scikit-learn xgboost tensorflow
```

4. Run the notebook step by step. It will:
   - Load all models
   - Take sample inputs
   - Predict across the 6 models
   - Generate a unified report using a fine-tuned GPT model (Model 7)

---

### 📦 Datasets Used

All datasets used for training the models are publicly available:

- [FER-2013 (Facial Expressions)](https://www.kaggle.com/datasets/msambare/fer2013/data)
- [Speech Emotion Features](https://www.kaggle.com/datasets/taqiyyaghazi/emotion-speech-features-for-speech-recognition)
- [Vital Signs Dataset](https://www.kaggle.com/datasets/nasirayub2/human-vital-sign-dataset?select=human_vital_signs_dataset_2024.csv)
- [Sleep & Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
- [Bone Fracture Risk Dataset](https://www.kaggle.com/datasets/ukveteran/fracture-risk-data)
- [Mental Health Sentiment Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/code)

> Model 7 is a GPT-based pre-trained model fine-tuned to generate health summaries from the 6 model outputs. It is not trained from scratch.
