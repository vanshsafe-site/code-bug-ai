"""
CodeBug AI -- Gradio demo
Loads the trained TF-IDF + Logistic Regression baseline and exposes a
simple textbox UI for pasting code and getting a prediction.
"""

import sys
import os
import spaces

@spaces.GPU(duration=1)
def _dummy_gpu_warmup():
    return True
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference import predict_bug

import gradio as gr

DISCLAIMER = (
    "⚠️ This model predicts the *likelihood* of a potential bug based on "
    "statistical patterns learned from real C functions (FFmpeg/Qemu, "
    "Devign dataset). It is **not a guaranteed static analyzer** and may "
    "not generalize well to other languages or small/synthetic code "
    "snippets. Use it as a signal, not a verdict."
)


def run_prediction(code):
    if not code or not code.strip():
        return "Please paste some code first.", ""
    result = predict_bug(code)
    label = result["prediction"]
    confidence = result["confidence"]
    return f"Prediction: {label}", f"Confidence: {confidence:.2%}"


with gr.Blocks(title="CodeBug AI") as demo:
    gr.Markdown("# 🐛 CodeBug AI")
    gr.Markdown(
        "A machine-learning system trained to detect potential bugs in C source code."
    )
    gr.Markdown(DISCLAIMER)

    code_input = gr.Textbox(
        label="Paste your C code here",
        placeholder="static int example(int a, int b) {\n    ...\n}",
        lines=15,
    )
    predict_btn = gr.Button("Analyze Code", variant="primary")

    prediction_output = gr.Textbox(label="Result")
    confidence_output = gr.Textbox(label="Model Confidence")

    predict_btn.click(
        fn=run_prediction,
        inputs=code_input,
        outputs=[prediction_output, confidence_output],
    )

if __name__ == "__main__":
    demo.launch()
