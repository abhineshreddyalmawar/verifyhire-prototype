from flask import Flask, render_template, request, redirect, url_for, session
from coordinator import process_candidates
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/")
def home():
    return render_template("index.html", results=None)

@app.route("/analyze", methods=["POST"])
def analyze():
    resume_files = request.files.getlist("resumes")
    job_description_text = request.form["job_description"]

    os.makedirs("uploads", exist_ok=True)
    resume_paths = []
    for f in resume_files:
        path = os.path.join("uploads", f.filename)
        f.save(path)
        resume_paths.append(path)

    results = process_candidates(resume_paths, job_description_text)

    session["results"] = results
    return redirect(url_for("show_results"))

@app.route("/results")
def show_results():
    results = session.get("results")
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
    