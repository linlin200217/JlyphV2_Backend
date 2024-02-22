import openai
import csv
import pandas as pd
from openai import OpenAI
from typing import Union, List, Dict, Optional, Tuple

DATAPATH = "./data"
os.makedirs(DATAPATH, exist_ok=True)
OPENAI_API_KEY = "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"

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

def data_process(data_path: str) -> Dict:
    data = pd.read_csv(data_path)
    data_title = data_path.split("/")[-1].split(".")[0]
    struct = {"data_title": data_title, "Categorical": {}, "Numerical": {}}
    for column in data.columns:
        if data[column].dtype.name == "object":
            struct["Categorical"][column] = data[column].tolist()
        else:
            struct["Numerical"][column] = data[column].tolist()
    return struct

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
    api_key = OPENAI_API_KEY,
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
