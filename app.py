import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from utils import DATAPATH, IMAGE_RESOURCE_PATH, data_process, image_recommendation, extract_mask_image
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
        widget: {
            x: number,
            y: number,
            width: number,
            height: number
            },
        image_id: string,
        mask_refine: number, (0, 1, 2; 2 by default)
    }
    RETURN DATA {
        mask_image_id: string,
    }
    """
    data = request.json
    widget = data.get('widget')  # 从请求中获取 widget
    image_id = data.get('image_id')  # 从请求中获取 image_id
    mask_refine = data.get('mask_refine')  # 从请求中获取 mask_refine
    
    # 调用你的处理函数，并传递收到的参数
    mask_image_id = extract_mask_image(widget, image_id, mask_refine)
    
    # 将新的 image_id 返回给前端
    return jsonify({"mask_image_id": mask_image_id})
    return None
    
@app.route("/generate_element", methods=["POST"])
async def generate_element():
    """
    INPUT DATA{
        selection: [
                    {"Name": "widget_bottom_bread", "Meat": "widget_meet", "Bread": "widget_top_bread", "Vegetable": "widget_tomato"},
                    {"HealthyDeg": "widget_lettuce", "PopularDeg": "widget_meet"}
                    ]
        prompt: str
        mask: {
                'Name': [widget_top_bread,2],
                'Meat': [widget_meet,2],
                'Bread': [widget_bottom_bread,2],
                'Vegetable': [widget_tomato,2],
                'HealthyDeg': [widget_lettuce,2],
                'PopularDeg': [widget_meet,2]
                }
        chosen_image_id: str}
    RETURN DATA {
        rgba_images_by_category: dic
    }
    """
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(DATAPATH, secure_filename(f.filename))
        f.save(file_path)
        data = request.json
        selection = data.get("selection")
        prompt = data.get("prompt")
        mask = data.get("mask")
        chosen_image_id = data.get("chosen_image_id")
        rgba_images_by_category = convert_RGBA_batch(selection, prompt, mask, chosen_image_id, file_path)
        return jsonify({"rgba_images_by_category": rgba_images_by_category})
    
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

