import os
import threading
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template_string

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Job tracking ───────────────────────────────────────────────────────────────
jobs = {}  # job_id -> {"status": "...", "output": "...", "error": "..."}

# ── Lazy-load the heavy ML pipeline so Flask starts instantly ──────────────────
_generator = None
_generator_lock = threading.Lock()


def get_generator():
    global _generator
    if _generator is None:
        with _generator_lock:
            if _generator is None:
                # Import here so Flask can start without waiting for torch/bert/whisper
                from Sem_4_VideoHighlightGenerator import UniversalHighlightGenerator
                _generator = UniversalHighlightGenerator(model_path=os.path.join(BASE_DIR, "highlight_model.pth"))
    return _generator


# ── Background worker ──────────────────────────────────────────────────────────
def run_highlight_job(job_id: str, input_path: str, output_path: str, sensitivity: float):
    try:
        jobs[job_id]["status"] = "loading_model"
        gen = get_generator()

        jobs[job_id]["status"] = "processing"
        gen.generate_highlights(input_path, output_path, sensitivity_std=sensitivity)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["output"] = os.path.basename(output_path)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save upload
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    filename = f"input_{uuid.uuid4().hex}{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # Create job
    job_id = uuid.uuid4().hex
    output_name = f"highlight_{job_id}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    sensitivity = float(request.form.get("sensitivity", 0.8))

    jobs[job_id] = {"status": "queued", "output": None, "error": None}

    thread = threading.Thread(
        target=run_highlight_job,
        args=(job_id, input_path, output_path, sensitivity),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    print("=" * 55)
    print("  Video Highlight Generator — Web App")
    print("  Open  http://localhost:5000  in your browser")
    print("=" * 55)
    app.run(debug=False, port=5000)
