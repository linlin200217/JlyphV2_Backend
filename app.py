import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from utils import DATAPATH, IMAGE_RESOURCE_PATH, COLOR, data_process, image_recommendation, extract_mask_image, convert_RGBA_batch, \
regenerate_rgb, defalt_layer_forexample, final_output_image, final_image_output_fordata
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

data_file_path = ''

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
        global data_file_path
        data_file_path = file_path
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
        prompt:str,
        mask_forall:{
            "Colname": str,
            "Widget": dic,
            "Refine_num": num,
            "Class": str("Categorical"/"Numerical")
        }
        chosen_image_id: str
    }
    RETURN DATA{
    rgba_images_by_category:dic
    }
    """

    data = request.json
    prompt = data.get("prompt")
    mask_forall = data.get("mask_forall")
    chosen_image_id = data.get("chosen_image_id")
    rgba_images_by_category = convert_RGBA_batch(prompt, mask_forall, chosen_image_id, data_file_path)
    return jsonify({"rgba_images_by_category": rgba_images_by_category})
    return None

@app.route("/generate_numerical_element", methods=["POST"])
async def generate_numerical_element():
    """
    INPUT DATA{
        mask_forall:{
            "Colname": str,
            "Widget": dic,
            "Refine_num": num,
            "Class": str("Categorical"/"Numerical")
        }
        chosen_image_id: str
    }
    RETURN DATA{
    defalt_layer_forexample: [{
        "Colname": str,
        "Widget": dic,
        "Refine_num": num,
        "Class": str("Categorical"/"Numerical")
        "outlier_id": str,
        "Layer": int,
        "mask_bool": array
    },]
    }
    """

    data = request.json
    mask_forall = data.get("mask_forall")
    chosen_image_id = data.get("chosen_image_id")
    defalt_layer = defalt_layer_forexample(chosen_image_id, mask_forall)
    return jsonify({"defalt_layer_forexample": defalt_layer})
    return None

@app.route("/generate_example", methods=["POST"])
async def generate_example():
    """
    INPUT DATA{
        dic_array:{
            "Colname": str,
            "widget": dic,
            "Refine_num": num,
            "Class": str("Categorical"/"Numerical"),
            "outlier_id": str,
            "mask_bool": array,
            "Layer": int,
            "Position": int,
            "Form": "Size"/'Number_Vertical','Number_Horizontal','Number_Path',None,
            "Gap": int,None,
        }
        image_id: str
    }
    RETURN DATA{
    generate_image_id: str
    }
    """
    data = request.json
    dic_array = data.get("dic_array")
    image_id = data.get("image_id")
    example = final_output_image(image_id,dic_array)
    return jsonify({"example": example})
    return None


@app.route("/final_generation", methods=["POST"])
async def final_generation():
    """
    INPUT DATA{
        result: dic,
        dic_array:{
            "Colname": str,
            "widget": dic,
            "Refine_num": num,
            "Class": str("Categorical"/"Numerical"),
            "rgba_id": None,
            "mask_bool": array,
            "Layer": int,
            "Position": int,
            "Form": "Size"/'Number_Vertical','Number_Horizontal','Number_Path',None,
            "Gap": int,None,
            "Path_Col": str, None,
        },
        image_id: str
    }
    RETURN DATA{
        final_generation_result:{
        data_index: image_id
    }
    """
    data = request.json
    result = data.get("result")
    dic_array = data.get("dic_array")
    image_id = data.get("image_id")
    final_generation_result = final_image_output_fordata(dic_array, result, data_file_path, image_id)
    return jsonify({"efinal_generation_result": final_generation_result})
    return None



@app.route("/regenerate", methods=["POST"])
def regenerate():
    """
    INPUT DATA{
        image_id:str,
        mask:{
            "Colname": str,
            "Widget": dic,
            "Refine_num": num
        }
        prompt?: str,
        whole_prompt?: str
    }
    RETURN DATA{
        re_generate_rgba_id : str
    }
    """
    data = request.json
    image_id = data.get("image_id")
    mask = data.get("mask")
    prompt = data.get("prompt")
    whole_prompt = data.get("whole_prompt")
    re_generate_rgba_id = regenerate_rgb(image_id, mask, prompt, whole_prompt)
    return jsonify({"re_generate_rgba_id": re_generate_rgba_id})

@app.route('/color', methods=["POST"])
def get_color():
    """
    POST
    {
        exist_color: list[str]
    }
    return
    {
        color: list[str]
    }
    """
    exist_color = request.json["exist_color"]
    return jsonify({"color": set(COLOR) - set(exist_color)})

@app.route("/image/<image_id>")
def get_image(image_id):
    """
    GET: HOST/image/<image_id> 
    return:
        image
    """
    return send_file(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9009, debug=True)


