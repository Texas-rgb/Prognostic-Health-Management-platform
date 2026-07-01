import json
import gradio as gr
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# 1. Custom Attention Layer
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name="att_weight",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.u = self.add_weight(
            name="att_context",
            shape=(feature_dim,),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        return tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        # Extend here if you add custom __init__ params in the future
        return config


# 2. Health state helper
def health_state(rul: float) -> str:
    if rul > 100:
        return "healthy"
    if rul > 40:
        return "warning"
    return "critical"


# 3. Load model
print("Loading model...")
model = tf.keras.models.load_model(
    "attention_lstm_model.keras",
    custom_objects={"AttentionLayer": AttentionLayer}
)

# Warmup — first TF call is always slow; do it now so users don't feel it
print("Warming up model...")
model.predict(np.zeros((1, 30, 21)), verbose=0)
print("Model ready.")


# 4. Inference with Monte Carlo Dropout for real confidence estimation
def predict(input_csv: str) -> str:
    # Input validation
    input_csv = input_csv.strip()
    if not input_csv:
        return json.dumps({"error": "No input provided"})

    try:
        raw_data = np.fromstring(input_csv, sep=",")
    except Exception:
        return json.dumps({"error": "Could not parse input. Make sure values are comma-separated numbers."})

    if raw_data.size != 630:
        return json.dumps({"error": f"Expected 630 values, got {raw_data.size}"})

    try:
        model_input = raw_data.reshape(1, 30, 21)

        # Monte Carlo Dropout — run 20 stochastic forward passes
        # training=True keeps dropout active at inference time
        mc_predictions = np.array([
            model(model_input, training=True).numpy()[0][0]
            for _ in range(20)
        ])

        rul_value = float(max(0.0, mc_predictions.mean()))
        std = mc_predictions.std()

        # Confidence: high when std is low relative to predicted RUL
        # Clipped to [0, 1] range
        raw_confidence = 1.0 - (std / (rul_value + 1e-5))
        confidence = round(float(np.clip(raw_confidence, 0.0, 1.0)) * 100, 1)

        return json.dumps({
            "rul": round(rul_value, 2),
            "confidence": confidence,
            "health": health_state(rul_value),
            "std": round(float(std), 2)
        })

    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {str(e)}"})


# 5. Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        placeholder="Paste 630 comma-separated sensor values here (30 cycles × 21 sensors)...",
        label="Sensor Input (CSV)"
    ),
    outputs=gr.Textbox(label="Prediction Result (JSON)"),
    title="Aviation Engine RUL Predictor",
    description=(
        "Predicts Remaining Useful Life (RUL) of aviation engines using an "
        "Attention-LSTM model trained on NASA CMAPSS FD001. "
        "Returns RUL in cycles, health state, and a confidence score derived from Monte Carlo Dropout."
    )
)

if __name__ == "__main__":
    demo.launch()