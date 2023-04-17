import json
import pandas as pd

def convert_json_to_csv(json_data):
    classification_data = []
    ner_data = []

    for item in json_data:
        text = item["text"]

        # Event Classification data
        classification_data.append({"text": text, "label": 1})

        # NER data
        for entity in item["entities"]:
            start = entity["start"]
            end = entity["end"]
            label = entity["label"]
            words = text[start:end].split()

            for idx, word in enumerate(words):
                if idx == 0:
                    ner_data.append({"sentence_id": len(classification_data), "word": word, "tag": f"B-{label}"})
                else:
                    ner_data.append({"sentence_id": len(classification_data), "word": word, "tag": f"I-{label}"})

    event_classification_df = pd.DataFrame(classification_data)
    ner_df = pd.DataFrame(ner_data)

    event_classification_df.to_csv("event_classification.csv", index=False)
    ner_df.to_csv("ner.csv", index=False)

json_data = []
# Load the data
with open('train.jsonl', 'r') as f:
    examples = [json_data.append(json.loads(line)) for line in f]

convert_json_to_csv(json_data)
