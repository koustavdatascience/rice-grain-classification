"""Lightweight Vercel entry point for the public GrainKind interface.

The full TensorFlow model is intentionally not imported here. The checked-in model file is
currently a Git LFS pointer, and serverless deployments should not attempt model inference
until a deployable artifact and dedicated inference runtime are available.
"""
from pathlib import Path

from flask import Flask, request, render_template

PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
    static_url_path="/static",
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return render_template(
            "index.html",
            prediction=None,
            filename=None,
            confidence=None,
            error="Live prediction is not enabled in this Vercel preview. Run the full Flask project locally after downloading the trained model with Git LFS.",
        )
    return render_template(
        "index.html",
        prediction=None,
        filename=None,
        confidence=None,
        error=None,
    )
