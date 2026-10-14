# 🩺 Kidney Health Analyzer & Prediction System

An end-to-end Machine Learning and Deep Learning system for Kidney disease prediction and medical image analysis, featuring a Gradio web application with Grad-CAM explainability and Docker support.

---

## 📁 Project Structure

```
├── analyti/                                # Web application & deep learning deployment
│   ├── app.py                              # Gradio web app with Grad-CAM heatmap visualization
│   ├── best_kidney_model_v(55.60).h5       # Deep learning model for scan classification
│   ├── Dockerfile                          # Containerization for cloud deployment
│   ├── requirements.txt                    # App dependencies
│   └── README.md                           # Web application deployment guide
├── datasets.csv                            # Clinical dataset for chronic kidney disease
├── feature_correlation_with_y_pred.csv     # Feature importance and correlation analysis
├── kidney_model.pkl                        # Trained Random Forest classifier
├── train.ipynb                             # Model training, preprocessing & evaluation notebook
├── finalmodel.ipynb                        # Final model pipeline and validation notebook
└── .gitignore                              # Git exclusion rules
```

---

## 🔬 Core Components

### 1. 🩻 Deep Learning Scan Classifier & Grad-CAM (`analyti/`)
- **Model:** Convolutional Neural Network trained to classify kidney CT and ultrasound scans into four categories:
  - `Cyst`
  - `Normal`
  - `Stone`
  - `Tumor`
- **Explainability:** Integrated **Grad-CAM** generates visual heatmaps that highlight the exact anatomical regions driving the AI's diagnosis.
- **Interface:** Interactive **Gradio** web dashboard with instant image upload and side-by-side diagnostic results.

#### Running the Web Application
```bash
cd analyti
pip install -r requirements.txt
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

#### Running via Docker
```bash
cd analyti
docker build -t kidney-analyzer .
docker run -p 7860:7860 kidney-analyzer
```

---

### 2. 📊 Tabular Kidney Disease Predictor (`train.ipynb` / `kidney_model.pkl`)
- **Algorithm:** Random Forest Classifier trained on clinical health indicators from `datasets.csv`.
- **Analysis:** Includes data cleaning, categorical encoding, feature scaling, and feature correlation analysis (`feature_correlation_with_y_pred.csv`).

---

## 🚀 Cloud Deployment
The `analyti/` package is ready for one-click deployment on platforms like:
- **Render** (via Dockerfile)
- **Railway** (via Dockerfile)
- **Hugging Face Spaces** (via Gradio / Docker SDK)

---

## 📜 License
This project is for educational and research purposes.
