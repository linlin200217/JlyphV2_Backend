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