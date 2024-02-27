import datetime
import random
import requests
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

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_KEY = os.environ.get("OPENAI_KEY")
DEVICE = "cuda"
MODEL_TYPE = "vit_h"
COLOR_MAP = {
    True: (255, 255, 255),
    False: (39, 78, 19),
}

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
