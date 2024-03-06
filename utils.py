import datetime
import random
import requests
import torch
import openai
import csv
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from openai import OpenAI
from scipy.optimize import curve_fit
from PIL import Image, ImageDraw
from typing import Union, List, Dict, Optional, Tuple
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
import os
from os.path import join, dirname
from dotenv import load_dotenv

# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.3'
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_KEY = os.environ.get("OPENAI_KEY")
DEVICE = "cuda"
MODEL_TYPE = "vit_h"
COLOR_MAP = {
    True: (255, 255, 255),
    False: (39, 78, 19),
}
COLOR = ['red',
         'orange',
         'yellow',
         'green',
         'blue',
         'indigo',
         'violet',
         'purple',
         'pink',
         'black',
         'white',
         'gray',
         'brown',
         'golden',
         'silver']

DATAPATH = "./data"
IMAGE_RESOURCE_PATH = "./resources"
WEIGHTSPATH = "./weights"
os.makedirs(DATAPATH, exist_ok=True)
os.makedirs(IMAGE_RESOURCE_PATH, exist_ok=True)
os.makedirs(WEIGHTSPATH, exist_ok=True)

# wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights
CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

PIPE_SDCC = StableDiffusionControlNetPipeline.from_pretrained(
    "/home/newdisk/Models/stable-diffusion-v1-5", controlnet=ControlNetModel.from_pretrained("/home/newdisk/Models/sd-controlnet-canny", torch_dtype=torch.float16), torch_dtype=torch.float16
).to(DEVICE)

def save_image(img: Image.Image, prefix: Optional[str], type: str = "png") -> str:
    img_copy = img.copy()
    img_resized = img_copy.resize((512, 512))
    prefix = prefix or "main_"
    image_id = int(datetime.datetime.now().timestamp() +
                   random.randrange(1, 10000))
    filename = f"{prefix}{image_id}.{type}"
    img_resized.save(os.path.join(IMAGE_RESOURCE_PATH, filename), format=type)
    return prefix + str(image_id)

def get_image_by_id(image_id: str) -> Image.Image:
    return Image.open(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png")).resize((512,512))

def csv_to_text(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    text_content = "Here is the data from the CSV file:\n"
    for row in data:
        row_text = ', '.join([f"{key}: {value}" for key, value in row.items()])
        text_content += f"- {row_text}\n"
    return text_content

def word_recommendation(system_prompt, data_path):
    client = OpenAI(
    api_key = OPENAI_KEY,
    )
    chat_completion = client.chat.completions.create(
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": csv_to_text(data_path)}
        ],
        model="gpt-4",
    )
    word_cloud_str = chat_completion.choices[0].message.content
    word_cloud_lst = [item.strip() for item in word_cloud_str.split(',')]
    return word_cloud_lst

def data_process(data_path: str) -> Dict:
    system_prompt = ("You will be provided with a csv data file. Analyse the data. First figure out the data subject, and based on the subject, associate several relevent objects.\n"
                "Requirement: At least generate 8 objects. They MUST BE all nouns which are highly pertinent to the data. They MUST BE physical objects that can be portrayed in a picture. DO NOT simply copy the attributes or values in the data.\n"
                "Example:\n"
                "User Input: Here is the data from the CSV file:\n"
                "- Name: McDonald, Meat: Beef, Bread: White, Vegetable: Tomato, Healthy Deg: 1, Popular Deg: 3\n"
                "- Name: KFC, Meat: Chicken, Bread: Wheat, Vegetable: Cucumber, Healthy Deg: 3, Popular Deg: 2\n"
                "- Name: BurgerKing, Meat: Beef, Bread: Wheat, Vegetable: Cucumber, Healthy Deg: 3, Popular Deg: 1\n"
                "- Name: KFC, Meat: Chicken, Bread: Wheat, Vegetable: Tomato, Healthy Deg: 4, Popular Deg: 2\n"
                "- Name: BurgerKing, Meat: Beef, Bread: Wheat, Vegetable: Tomato, Healthy Deg: 2, Popular Deg: 1\n"
                "- Name: McDonald, Meat: Beef, Bread: Honey, Vegetable: Cucumber, Healthy Deg: 1, Popular Deg: 3\n"
                "Your answer:\n"
                "hamburger, sandwich, chip, fried chicken, potato, tomato, pizza, restaurant"
)
    data = pd.read_csv(data_path)
    data_title = data_path.split("/")[-1].split(".")[0]
    struct = {"data_title": data_title, "Categorical": {}, "Numerical": {}, "Wordcloud": {}}
    for column in data.columns:
        if data[column].dtype.name == "object":
            struct["Categorical"][column] = data[column].tolist()
        else:
            struct["Numerical"][column] = data[column].tolist()
    struct["Wordcloud"] = word_recommendation(system_prompt, data_path)
    return struct

def image_recommendation(user_prompt):
    ideation_image_set = []

    prompt = f"A single, cartoon style, 2D {user_prompt}, flat with no shadow, white background"

    client = OpenAI(
        api_key=OPENAI_KEY
    )

    image_generation = client.images.generate(
        prompt=prompt,
        model="dall-e-2",
        n=3, 
        size="1024x1024"
    )
    for i, image in enumerate(image_generation.data):
        response = requests.get(image.url)
        if response.status_code == 200:
            image_bytes_io = BytesIO(response.content)
            image = Image.open(image_bytes_io)
            ideation_image_id = save_image(image, "ideal")
            ideation_image_set.append(ideation_image_id)
        else:
            print("Failed to retrieve image.")
    return ideation_image_set

def extract_mask(widget, image_id:str, mask_refine:int):
  box = widget
  box = np.array([
      box['x'],
      box['y'],
      box['x'] + box['width'],
      box['y'] + box['height']
  ])
  image_bgr = cv2.imread(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  mask_predictor.set_image(image_rgb)
  masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
  )
  mask = masks[mask_refine]
  return mask

def extract_initial_image(widget, image_id:str, mask_refine:int):
  image_bgr = cv2.imread(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  mask = extract_mask(widget, image_id, mask_refine)
  width = len(mask)
  height = len(mask)
  image = Image.new("RGB", (width, height))

  mask_shape = mask.shape
  initial_image = Image.new('RGB', (mask_shape[1], mask_shape[0]), (255, 255, 255))
  for i in range(mask_shape[0]):
      for j in range(mask_shape[1]):
          if mask[i][j]:
              initial_image.putpixel((j, i), tuple(image_rgb[i][j]))
  out_initial_id = save_image(initial_image, "initial")
  return out_initial_id

def extract_mask_image(widget, image_id:str, mask_refine:int):
  image_bgr = cv2.imread(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  mask = extract_mask(widget, image_id, mask_refine)
  width = len(mask)
  height = len(mask)
  image = Image.new("RGB", (width, height))
  for y in range(height):
      for x in range(width):
          pixel_color = COLOR_MAP[mask[y][x]]
          image.putpixel((x, y), pixel_color)
  out_mask_id = save_image(image, "mask")

  outlier_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
  draw = ImageDraw.Draw(outlier_image)
  for y in range(height):
    for x in range(width):
        if mask[y][x]:
            pixel_color = (0, 0, 0, 0)
            if (
                x == 0
                or x == width - 1
                or y == 0
                or y == height - 1
                or not mask[y - 1][x]
                or not mask[y + 1][x]
                or not mask[y][x - 1]
                or not mask[y][x + 1]
            ):
                for i in range(15):
                    if x + i < width:
                        draw.point((x + i, y), fill=(39, 78, 19))
  out_outlier_id = save_image(outlier_image, "outlier")
  mask_result = out_mask_id
  return mask_result


###### FORMAL ########
def make_prompt_for_each_mask(prompts: List[str], cat_num, path) -> Dict[str, List[Tuple]]:

    df = pd.read_csv(path)
    dic_cat_num = {"Categorical": [], "Numerical": []}

    widgets_str_0 = {k: str(v) for k, v in cat_num[0].items()}
    widgets_str_1 = {k: str(v) for k, v in cat_num[1].items()}

    reverse_widgets_str_1 = {v: k for k, v in widgets_str_1.items()}

    for key, widget_str in widgets_str_0.items():
        dic_cat_num["Categorical"].append(key)

        if widget_str in reverse_widgets_str_1:
            reverse_widgets_str_1.pop(widget_str)  

    for key in reverse_widgets_str_1.values():
        dic_cat_num["Numerical"].append(key)

    prompts_dict = {}
    prompt_index = 0

    for category, items in dic_cat_num.items():
        for item in items:
            if category == "Categorical" and prompt_index < len(prompts):
                item_duplicated = df[item].drop_duplicates()
                num = len(item_duplicated)

                colors = random.sample(COLOR, min(num, len(COLOR)))
                prompt_list = [(f"A {color} {prompts[prompt_index]}", color, value) for value, color in zip(item_duplicated, colors)]
                prompts_dict[item] = prompt_list
                prompt_index += 1

            elif category == "Numerical" and prompt_index < len(prompts):
                color = random.choice(COLOR)
                prompt_list = [(f"A {color} {prompts[prompt_index]}", color, item)]
                prompts_dict[item] = prompt_list
                prompt_index += 1
    if prompt_index != len(prompts):
        raise ValueError("Not all prompts have been used. Please check the categoricals_numericals dictionary and prompts list for consistency.")

    return prompts_dict



def generate_image(prompt: str, image_id: str, image_prefix: Optional[str] = None):
    img = np.array(get_image_by_id(image_id))
    low_threshold = 100
    high_threshold = 200
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    canny_image = Image.fromarray(img).resize((512,512))

    out = PIPE_SDCC(prompt, num_inference_steps=20,
                    image=canny_image).images[0].resize((512,512))
    out_image_id = save_image(out, "generate")
    return out_image_id


def generate_images_by_category(category_prompts, category_image_ids):
    generated_images = {}
    for category, prompts in category_prompts.items():
        image_id = category_image_ids[category]  # 获取当前类别对应的image_id
        out_image_ids = []
        for prompt, _, _ in prompts:
            out_image_id = generate_image(prompt, image_id)  # 生成图片并获取图片ID
            out_image_ids.append(out_image_id)
        generated_images[category] = out_image_ids
    return generated_images
  
####FORMAL####
def convert_RGBA_batch(prompt, mask_forall, chosen_image_id, path):
    df = pd.read_csv(path)
    mask_ori = {item["Colname"]: [item["Widget"], item["Refine_num"]] for item in mask_forall}
    categorical_dict = {}
    numerical_dict = {}
    for item in mask_forall:
        if item["Class"] == "Categorical":
            categorical_dict[item["Colname"]] = item["Widget"]
        elif item["Class"] == "Numerical":
            numerical_dict[item["Colname"]] = item["Widget"]
    selection = [categorical_dict, numerical_dict]
    categoricals = []
    rgba_images_by_category = {}
    initial_image_ids = {}
    category_image_ids = {}
    dic_cat_num = {"Categorical": [], "Numerical": []}

    widgets_str_0 = {k: str(v) for k, v in selection[0].items()}
    widgets_str_1 = {k: str(v) for k, v in selection[1].items()}


    reverse_widgets_str_1 = {v: k for k, v in widgets_str_1.items()}


    for key, widget_str in widgets_str_0.items():
        dic_cat_num["Categorical"].append(key)
        if widget_str in reverse_widgets_str_1:
            reverse_widgets_str_1.pop(widget_str) 


    for key in reverse_widgets_str_1.values():
        dic_cat_num["Numerical"].append(key)

    keys_to_keep = set(dic_cat_num["Categorical"] + dic_cat_num["Numerical"])
    masks = {key: value for key, value in mask_ori.items() if key in keys_to_keep}
    ###### Prompts要改哦
    masks_length = len(masks.keys())
    prompts = [prompt] * masks_length
    for c in masks.keys():
        initial_widget = masks[c][0]
        initial_mask_num = masks[c][1]
        initial_image = extract_initial_image(initial_widget, chosen_image_id, initial_mask_num)  # 假设这返回一个字符串
        initial_image_ids[c] = initial_image
    category_image_ids = initial_image_ids

    category_prompts = make_prompt_for_each_mask(prompts, selection, path)
    generate_image_ids = generate_images_by_category(category_prompts, category_image_ids)

    for category, ids in generate_image_ids.items():
        mask_widget = masks[category][0]
        mask_num = masks[category][1]
        mask = extract_mask(mask_widget, chosen_image_id, mask_num)
        mask_np = np.array(mask)
        mask_8bit = mask_np.astype(np.uint8) * 255

        rgba_image_details = []
        for idx, image_id in enumerate(ids):
            new_image = get_image_by_id(image_id)
            image_np = np.array(new_image)
            extracted = cv2.bitwise_and(image_np, image_np, mask=mask_8bit)
            alpha = np.zeros_like(mask_8bit)
            alpha[mask_np] = 255
            extracted_rgba = cv2.cvtColor(extracted, cv2.COLOR_BGR2BGRA)
            extracted_rgba[:, :, 3] = alpha
            image_rgba = Image.fromarray(extracted_rgba).resize((512, 512))
            rgba_image_id = save_image(image_rgba, "rgba")
            prompt_detail = category_prompts[category][idx][2]
            rgba_image_details.append({'rgba_image_id': rgba_image_id, 'prompt_detail': prompt_detail})

        rgba_images_by_category[category] = rgba_image_details

    return rgba_images_by_category


def regenerate_prompt(prompt: Optional[str] = None, whole_prompt: Optional[str] = None):
    color = random.choice(COLOR)
    if prompt and whole_prompt:
        raise ValueError("prompt and whole_prompt cannot both be provided.")
    elif prompt:
        return f"A {color} {prompt}."
    elif whole_prompt:
        return whole_prompt
    else:
        raise ValueError("Either prompt or whole_prompt must be provided.")
    
def regenerate_wholeimage(image_id: str,prompt: Optional[str] = None, whole_prompt: Optional[str] = None):
    prompt_re = regenerate_prompt(prompt, whole_prompt)
    re_generate_image_id = generate_image(prompt_re, image_id)
    return re_generate_image_id

def convert_RGBA(image_id, mask):
    new_image = get_image_by_id(image_id)
    image_np = np.array(new_image)
    mask_8bit = mask.astype(np.uint8) * 255
    extracted = cv2.bitwise_and(image_np, image_np, mask=mask_8bit)
    alpha = np.zeros_like(mask_8bit)
    alpha[mask] = 255
    extracted_rgba = cv2.cvtColor(extracted, cv2.COLOR_BGR2BGRA)
    extracted_rgba[:, :, 3] = alpha
    image_rgba = Image.fromarray(extracted_rgba).resize((512,512))
    rgba_image_id = save_image(image_rgba, "rgba")
    return rgba_image_id

def regenerate_rgb(image_id: str, mask, prompt: Optional[str] = None, whole_prompt: Optional[str] = None):
    mask_let_transformed = {mask["Colname"]: [mask["Widget"], mask["Refine_num"]]}
    values = [*mask_let_transformed.values()]
    mask_wid = values[0][0]
    mask_num = values[0][1]
    mask_ini = extract_initial_image(mask_wid, image_id, mask_num)
    re_generate_image_id = regenerate_wholeimage(mask_ini, prompt, whole_prompt)
    mask_re = extract_mask(mask_wid, image_id, mask_num)
    re_generate_rgba_id = convert_RGBA(re_generate_image_id, mask_re)
    return re_generate_rgba_id

def are_widgets_equal(widget_a, widget_b):
    return widget_a['x'] == widget_b['x'] and \
           widget_a['y'] == widget_b['y'] and \
           widget_a['width'] == widget_b['width'] and \
           widget_a['height'] == widget_b['height']

def Extract_Numerical_dic(dic_array):
    processed = []
    widget_seen = {}

    for entry in dic_array:
        widget = tuple(entry['Widget'].items()) 
        if widget in widget_seen:
            if entry['Class'] == 'Numerical':
                processed = [e for e in processed if not are_widgets_equal(e['Widget'], entry['Widget'])]
                processed.append(entry)
        else:
            widget_seen[widget] = True
            processed.append(entry)
    return processed

def extract_outlier_image(widget, image_id:str, mask_refine:int):
  image_bgr = cv2.imread(os.path.join(IMAGE_RESOURCE_PATH, image_id + ".png"))
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  mask = extract_mask(widget, image_id, mask_refine)
  width = len(mask)
  height = len(mask)

  outlier_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
  draw = ImageDraw.Draw(outlier_image)
  for y in range(height):
    for x in range(width):
        if mask[y][x]:
            pixel_color = (0, 0, 0, 0)
            if (
                x == 0
                or x == width - 1
                or y == 0
                or y == height - 1
                or not mask[y - 1][x]
                or not mask[y + 1][x]
                or not mask[y][x - 1]
                or not mask[y][x + 1]
            ):
                for i in range(15):
                    if x + i < width:
                        draw.point((x + i, y), fill=(39, 78, 19))
  outlier_image = outlier_image.resize((512,512))
  out_outlier_id = save_image(outlier_image, "outlier")
  return out_outlier_id

def find_outlier_forexample(image_id, dic_array):
    for item in Extract_Numerical_dic(dic_array):
        widget = item['Widget']
        refine_num = item['Refine_num']
        outlier_id = extract_outlier_image(widget,image_id,refine_num)
        item['outlier_id'] = outlier_id
    
    return Extract_Numerical_dic(dic_array)

def defalt_layer_forexample(image_id, dic_array):
  dic_outlier = find_outlier_forexample(image_id, dic_array)
  grouped_by_y = {}
  for item in dic_outlier:
      y_key = item["Widget"]["y"]
      if y_key not in grouped_by_y:
          grouped_by_y[y_key] = []
      grouped_by_y[y_key].append(item)
  sorted_items = []
  for items in grouped_by_y.values():
      sorted_items.extend(sorted(items, key=lambda x: -(x["Widget"]["y"] + x["Widget"]["width"])))


  current_layer = 1
  previous_item = None
  for item in sorted_items:
      if previous_item and (item["Widget"]["y"] != previous_item["Widget"]["y"] or (item["Widget"]["y"] + item["Widget"]["width"] != previous_item["Widget"]["y"] + previous_item["Widget"]["width"])):
          current_layer += 1
      item["Layer"] = current_layer
      previous_item = item


  sorted_dic_array = sorted(dic_outlier, key=lambda x: x["Layer"])
  for dic in sorted_dic_array:
    widget = dic['Widget']
    refine_num = dic['Refine_num']
    dic['mask_bool'] = extract_mask(widget, image_id, refine_num)
  return sorted_dic_array


def determine_form(dic_list):
    unique_forms = set()
    for dic in dic_list:
        form = dic.get('Form')
        if form: 
            unique_forms.add(form)
    combinations = {
        frozenset(['Size']): "Size",
        frozenset(['Number_Vertical']): "Number_Vertical",
        frozenset(['Number_Horizontal']): "Number_Horizontal",
        frozenset(['Number_Path']): "Number_Path",
        frozenset(['Size', 'Number_Vertical']): "Size_Number_Vertical",
        frozenset(['Size', 'Number_Horizontal']): "Size_Number_Horizontal",
        frozenset(['Size', 'Number_Path']): "Size_Number_Path",
    }
    return combinations.get(frozenset(unique_forms), "Undefined Combination")

def vertical_position(dic_list):
    sorted_dic_list = sorted(dic_list, key=lambda x: (x['widget']['y'], -(x['widget']['y'] - x['widget']['height'])), reverse=True)
    for index, dic in enumerate(sorted_dic_list, start=1):
        dic['Position'] = index
    return sorted_dic_list

def horizontal_position(dic_list):
    sorted_dic_list = sorted(
        dic_list, 
        key=lambda x: (x['widget']['x'], -(x['widget']['x'] + x['widget']['width']))
    )
    for index, dic in enumerate(sorted_dic_list, start=1):
        dic['Position'] = index
    
    return sorted_dic_list

def form_with_position(dic_list):
    form = determine_form(dic_list)
    if form in ["Size", "Number_Vertical", "Number_Path", "Size_Number_Vertical", "Size_Number_Path"]:
        return vertical_position(dic_list)
    elif form in ["Number_Horizontal", "Size_Number_Horizontal"]:
        return horizontal_position(dic_list)
    else:
        return [dic_list]
    
def get_contour_center(contour):
    """计算轮廓的中心坐标"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def pre_dic_forexample(dic_array):
    dic_array_update = form_with_position(dic_array)
    summary_dic = {
        "move": [],
        "num": [],
        "gap": [],
        "direction": []
    }

    sorted_dic_list = sorted(dic_array_update, key=lambda x: x['Position'])

    for dic in sorted_dic_list:
        form = dic.get('Form')
        layer = dic['Position']
        gap = dic.get('Gap', 0)  
        
        if form in ["Number_Vertical", "Number_Horizontal"]:
            direction = "vertical" if form == "Number_Vertical" else "horizontal"
            summary_dic["move"].append(layer - 1)
            summary_dic["num"].append(2)  
            summary_dic["gap"].append(gap)
            summary_dic["direction"].append(direction)

    return summary_dic

def dic_forexample(dic_array):
  predic = pre_dic_forexample(dic_array)
  move = predic["move"]
  num = predic["num"]
  gap = predic["gap"]
  directions = predic["direction"]

  dic = {}
  num = [1 if i not in move else num[move.index(i)] for i in range(len(dic_array))]
  gap = [0 if i not in move else gap[move.index(i)] for i in range(len(dic_array))]
  directions = ["" if i not in move else directions[move.index(i)] for i in range(len(dic_array))]

  x = 0
  y = 0

  for slice, (count, _gap, direction) in enumerate(zip(num, gap, directions)):
    dic[slice] = [[x,y]]
    for c in range(1, count):
        positions = dic.get(slice, [[x, y]])
        positions.append([x, y])
        if direction == "horizontal":
            x -=- _gap
            positions[-1][0] = x
        elif direction == "vertical":
            y -=- _gap
            positions[-1][1] = y
        dic[slice] = positions

  return dic

def Set_Size_Num_forexample(image_id,dic_array):
  images = []
  masks = []
  order = []
  scale_factors = []
  processed_images = []
  processed_masks = []
  image_id = image_id
  sorted_data = sorted(form_with_position(dic_array), key=lambda x: x['Position'])
  dic_for_pos = dic_forexample(dic_array)
  for item in sorted_data:
    images.append(item['outlier_id'])
    masks.append(item['mask_bool'])
    order.append(item['Layer'])
    scale_factors.append(1.2 if item['Form'] == 'Size' else 1)
  canvas_size = (512,512,3)
  canvas = np.ones(canvas_size, dtype=np.uint8)*255
  for image_id, mask, scale_factor, (key, positions) in zip(images, masks, scale_factors, dic_for_pos.items()):
    ori_image = get_image_by_id(image_id)
    ori_image_rgb = np.array(ori_image)
    image = cv2.cvtColor(ori_image_rgb, cv2.COLOR_RGB2BGR)
    original_mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_center = get_contour_center(largest_contour)

        new_width, new_height = int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(original_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        new_contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_contour_center = get_contour_center(max(new_contours, key=cv2.contourArea))

        center_dx, center_dy = new_contour_center[0] - contour_center[0], new_contour_center[1] - contour_center[1]
        M = np.float32([[1, 0, -center_dx], [0, 1, -center_dy]])

        corrected_scaled_mask = cv2.warpAffine(scaled_mask, M, (canvas_size[1], canvas_size[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        corrected_scaled_image = cv2.warpAffine(scaled_image, M, (canvas_size[1], canvas_size[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        for dx, dy in positions:
            M = np.float32([[1, 0, dx], [0, 1, -dy]])
            translated_mask = cv2.warpAffine(corrected_scaled_mask, M, (canvas_size[1], canvas_size[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            translated_image = cv2.warpAffine(corrected_scaled_image, M, (canvas_size[1], canvas_size[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            

            processed_images.append((key, translated_image, translated_mask))
  for layer in order:
    layer_index = layer - 1 
    for item in processed_images:
        if item[0] == layer_index:
            image, mask = item[1], item[2]

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            canvas[mask > 0] = image_rgb[mask > 0]

  final_image = Image.fromarray(canvas) 
  size_num_id = save_image(final_image, "Numerical")
  return size_num_id



def transform_NumpyToBoolean(dic):
    bool_dic = {}
    for key, mask in dic.items():
        bool_mask = mask != 0
        bool_dic[key] = bool_mask
    return bool_dic

def extract_colnameTopath(masks,dic_array):
  colname_to_mask = {}
  for index, dic in enumerate(dic_array):
      colname = dic['Colname'] 
      mask = masks[index]  
      colname_to_mask[colname] = mask
  return colname_to_mask 

def quadratic_curve(x, a, b, c):
    return a * x**2 + b * x + c
def extract_smoothed_path(mask):
    points = np.column_stack(np.where(mask))
    params, _ = curve_fit(quadratic_curve, points[:, 1], points[:, 0])
    x_fit = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100)
    y_fit = quadratic_curve(x_fit, *params)
    path_image = np.zeros_like(mask, dtype=np.uint8)
    path_coords = np.column_stack((y_fit, x_fit)).astype(int)
    for x, y in path_coords:
        if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]:
            path_image[x, y] = 255
    path_array = np.zeros_like(mask, dtype=bool)
    path_array[np.where(path_image == 255)] = True
    return path_array

def load_image_bgra(image_id):
    """Load an image from a path and convert it to a BGRA NumPy array."""
    with get_image_by_id(image_id) as img:
        rgba_image = img.convert('RGBA')
        rgba_array = np.array(rgba_image)
        bgra_array = rgba_array[..., [2, 1, 0, 3]]
        return bgra_array
def paste_with_alpha_bgra(target, source, start_x, start_y):
    """Manually paste a BGRA image onto another BGR image considering alpha transparency."""
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            sy = start_y + y
            sx = start_x + x
            if 0 <= sy < target.shape[0] and 0 <= sx < target.shape[1]:
                alpha = source[y, x, 3] / 255.0
                if alpha > 0:
                    target[sy, sx, :3] = (1 - alpha) * target[sy, sx, :3] + alpha * source[y, x, :3]
def find_bottom_baseline_bgra(extracted):
    """Find the bottommost non-transparent pixel row in the extracted BGRA image."""
    for i in range(extracted.shape[0] - 1, -1, -1):
        if np.any(extracted[i, :, 3] != 0):
            return i
    return None

def find_top_baseline_bgra(extracted):
    """Find the topmost non-transparent pixel row in the extracted BGRA image."""
    for i in range(extracted.shape[0]):
        if np.any(extracted[i, :, 3] != 0):
            return i
    return None

def segment_curve_and_paste_extracted_bgra(image_id, dic_array):
    image_ids = []
    masks = []
    gaps = []
    nums = []
    directons = []
    masks_array = transform_NumpyToBoolean(extract_colnameTopath(Set_Size_Num_forexample(image_id,dic_array)[0],dic_array))

    for item in dic_array:
      if 'Path' in (item.get('Form') or ''):
        image_ids.append(item['outlier_id'])
        gaps.append(item.get('Gap', None))
        if item['Path_Col'] and item['Path_Col'] in masks_array:
            masks.append(masks_array[item['Path_Col']])
    nums = [2] * len(image_ids)
    directions = ["bottom"] * len(image_ids)
    canvas_id = Set_Size_Num_forexample(image_id,dic_array)[1]

    canvas_pil = get_image_by_id(canvas_id)
    canvas_rgb = np.array(canvas_pil)
    canvas_bgr = canvas_rgb[:, :, ::-1]
    for mask, num, image_id, gap, direction in zip(masks, nums, image_ids, gaps, directions):
        points = np.argwhere(extract_smoothed_path(mask))
        if len(points) < num:
            raise ValueError("Not enough points on the curve to segment into n parts")

        segmented_points = points[np.round(np.linspace(0, len(points) - 1, num)).astype(int)]
        extracted = load_image_bgra(image_id)

        if direction == 'top':
            baseline = find_bottom_baseline_bgra(extracted)
            gap_sign = -1
        elif direction == 'bottom':
            baseline = find_top_baseline_bgra(extracted)
            gap_sign = 1
        else:
            raise ValueError("Invalid direction. Must be 'top' or 'bottom'.")

        if baseline is not None:
            for point in segmented_points:
                top_left_x = point[1] - extracted.shape[1] // 2
                top_left_y = point[0] - baseline + gap_sign * gap
                paste_with_alpha_bgra(canvas_bgr, extracted, top_left_x, top_left_y)
    final_rgb = canvas_bgr[:, :, ::-1]
    final_image_pil = Image.fromarray(final_rgb)
    numerical_path_image_id = save_image(final_image_pil, "Numerical")
    return numerical_path_image_id

def final_output_image(image_id, dic_array):
  form_combination = determine_form(dic_array)
  if form_combination in ["Number_Vertical", "Number_Horizontal", "Size_Number_Vertical", "Size_Number_Horizontal", "Size"]:
    return Set_Size_Num_forexample(image_id,dic_array)[1]
  elif form_combination in ["Number_Path", "Size_Number_Path"]:
    return segment_curve_and_paste_extracted_bgra(image_id, dic_array)
