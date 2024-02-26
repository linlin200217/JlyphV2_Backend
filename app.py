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

@app.route("/maskselect", methods=["POST"])
async def maskselect():
    """
    INPUT DATA{
        widget: {'x': 162, 'y': 279, 'width': 204, 'height': 58}
        image_id: str
    }
    RETURN DATA {
        mask_image_id:str,
        outlier_image_id:str
    }
    """
    return None

async def generate_element():
    """
    INPUT DATA{
        masks: {
            'Meat': [widget_meet,2], 
            'Vegetable': [widget_tomato,2],
            'Name': [widget_bottom_bread,2],
            'Bread': [widget_top_bread,2]
            }
        image_id: str
    }
    RETURN DATA {
        {'Meat':[{'rgba_image_id': 'rgba1708965745', 'prompt_detail': 'Beef'}, {'rgba_image_id': 'rgba1708963415', 'prompt_detail': 'Chicken'}],...}
    }
    """
    return None

@app.route("/image/<image_id>")
def get_image(image_id):
    """
    GET: HOST/image/<image_id> 
    return:
        image
    """
    return send_file(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9005, debug=True)

