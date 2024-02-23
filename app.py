from flask import Flask
from flask_cors import CORS
from utils import DATAPATH, IMAGE_RESOURCE_PATH, data_process, image_recommendation

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
    image_id = image_recommendation(request.json)
    return jsonify({"status": "success", "image_id": image_id})

