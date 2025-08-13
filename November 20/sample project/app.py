from flask import Flask, render_template, request
from text_to_speech_util import text_to_speech

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        text_to_speech(text)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
