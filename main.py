import json
from transformers import pipeline
import pandas as pd

if __name__ == '__main__':
    try:
        with open('raw_data.json', 'r') as file:
            data = json.load(file)
            data = data['conversation']
            data_size = len(data)
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")

    result_table = pd.DataFrame(columns=['step', 'speaker', 'text', 'sentiment', 'intent'])
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    # model="tasksource/deberta-base-long-nli", device=0 for GPU utilization

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