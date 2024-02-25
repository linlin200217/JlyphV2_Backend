import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import DATAPATH, IMAGE_RESOURCE_PATH, data_process, image_recommendation
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Backend listening...'

@app.route("/upload", methods=["POST"])
async def load_data():
    """
    POST FILE
    return json data {
        data_title: string,
        Categorical: [],
        Numerical: [],
        Wordcloud: []
    }
    """
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(DATAPATH, secure_filename(f.filename))
        f.save(file_path)
        struct = data_process(file_path)
        return jsonify(struct)

@app.route("/pregenerate", methods=["POST"])
async def pregenerate():
    """
    """
    image_ids = image_recommendation(request.json["user_prompt"])
    return jsonify({"status": "success", "image_id": image_ids})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9005, debug=True)

