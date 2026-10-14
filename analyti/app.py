# app.py
import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Load your trained model ---
MODEL_PATH = "best_kidney_model_v(55.60).h5"
base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- Build Grad-CAM model mapping input -> (conv2d_7, final_output) ---
input_tensor = tf.keras.Input(shape=(224, 224, 3))
x = input_tensor
conv_output = None
for layer in base_model.layers:
    x = layer(x)
    if layer.name == "conv2d_7":
        conv_output = x

grad_model = tf.keras.Model(inputs=input_tensor, outputs=[conv_output, x])

# --- Define class labels ---
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# --- Jet colormap for Grad-CAM overlay ---
def apply_jet_colormap(heatmap):
    # Heatmap is 2D float in [0, 1]
    r = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

def generate_gradcam(img_batch, pred_index):
    with tf.GradientTape() as tape:
        conv_features, predictions = grad_model(img_batch)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = conv_features[0] @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-10)

    cam_img = Image.fromarray((cam.numpy() * 255).astype(np.uint8)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    cam_norm = np.array(cam_img, dtype=np.float32) / 255.0
    return Image.fromarray(apply_jet_colormap(cam_norm))

# --- Prediction and Explainability function ---
def predict_kidney(img):
    if img is None:
        return None, None

    # Resize and normalize image
    orig_img = img.convert("RGB").resize((224, 224))
    img_array = np.array(orig_img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict class probabilities
    preds = base_model.predict(img_batch)[0]
    pred_idx = int(np.argmax(preds))
    confidences = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    # Generate Grad-CAM heatmap overlay
    heatmap_img = generate_gradcam(img_batch, pred_idx)
    overlay = Image.blend(orig_img, heatmap_img, alpha=0.45)

    return confidences, overlay

# --- Gradio interface ---
demo = gr.Interface(
    fn=predict_kidney,
    inputs=gr.Image(type="pil", label="Upload Kidney CT / Ultrasound Scan"),
    outputs=[
        gr.Label(num_top_classes=4, label="Predicted Condition & Confidence"),
        gr.Image(type="pil", label="Grad-CAM Explainability Heatmap")
    ],
    title="🩺 Kidney Health Analyzer + Grad-CAM",
    description="Upload a CT or Ultrasound image to classify the condition (Cyst, Normal, Stone, or Tumor) and visualize which regions triggered the AI prediction."
)

# --- Run app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share_link = os.environ.get("GRADIO_SHARE", "False").lower() in ("true", "1")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share_link)
