import json
import yaml
from transformers import pipeline
import pandas as pd
import importlib.resources
from pathlib import Path


def inference(classifier, input_text, text_labels):
    result = classifier(input_text, text_labels, multi_label=False)
    return result['sequence'], result['labels'][0]


def load_yaml(file_name):
    try:
        with open(file_name, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("The file was not found.")
    except yaml.YAMLError as e:
        print("Error decoding YAML.")


def load_json(file_name):
    try:
        with open('../raw_data.json', 'r') as file:
            return json.load(file)['conversation']
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")


if __name__ == '__main__':

    pretrained_model = load_yaml("../config.yaml")['model']['main']
    data = load_json("../raw_data.json")
    data_size = len(data)
    result_table = pd.DataFrame(columns=['step', 'speaker', 'text', 'sentiment', 'intent'])
    classifier = pipeline("zero-shot-classification", model=pretrained_model, device=0)
    # device=0 for GPU utilization

    sentiment_labels = ["very positive", "very negative", "neutral"]
    intent_labels = ["payment", "shipment", "model details", "pricing", "other"]

    for i in range(data_size):
        content = data[i]['text']
        text, sentiment_result = inference(classifier, content, sentiment_labels)
        text, intent_results = inference(classifier, content, intent_labels)
        temp_df = pd.DataFrame({'step': i + 1,
                                'speaker': data[i]['speaker'],
                                'text': text,
                                'sentiment': sentiment_result,
                                'intent': [intent_results]})
        result_table = result_table._append(temp_df, ignore_index=True)

    print(result_table)