# 🩺 Kidney Health Analyzer + Grad-CAM

A deep learning web application built with **TensorFlow / Keras** and **Gradio** to classify kidney CT and ultrasound scans into four diagnostic categories (`Cyst`, `Normal`, `Stone`, `Tumor`), complete with **Grad-CAM explainability heatmaps** highlighting the exact pathological regions driving the AI's diagnosis.

---

## 🚀 Deployment Options

### Option 1: Docker (Local or Any Cloud Server)

1. **Build Docker image:**
   ```bash
   docker build -t kidney-health-analyzer .
   ```

2. **Run container:**
   ```bash
   docker run -p 7860:7860 kidney-health-analyzer
   ```

3. Open **http://localhost:7860** in your browser.

---

### Option 2: Render (Free Web Service)

1. Push this folder to a GitHub repository.
2. Log into [Render](https://render.com) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Select **Docker** environment (Render will automatically detect `Dockerfile`).
5. Set:
   - **Plan:** Free
   - **Port:** `7860` (or leave default, Render sets `$PORT` automatically)
6. Click **Deploy Web Service**.

---

### Option 3: Railway

1. Push this folder to GitHub.
2. Go to [Railway.app](https://railway.app), click **New Project** -> **Deploy from GitHub repo**.
3. Select this repository. Railway automatically detects `Dockerfile` and deploys the container.

---

### Option 4: Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Docker** or **Gradio** as SDK.
3. Push `app.py`, `requirements.txt`, and `best_kidney_model_v(55.60).h5` to your Space repository.

---

## 🧪 Local Run (Without Docker)

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```
App will be accessible at: `http://localhost:7860`
