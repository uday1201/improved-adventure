import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline

# Load BERT pre-trained models
PRETRAINED_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
sequence_classifier = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
token_classifier = BertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=4)

# Custom label names for the classification task and NER task
CLASS_LABELS = ["no_event", "event"]
NER_LABELS = ["O", "B-DESC", "B-LOC", "B-DATE"]

# Function to perform event classification
def classify_event(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = sequence_classifier(**inputs)
    logits = outputs.logits.detach().numpy()
    probabilities = torch.softmax(outputs.logits, dim=-1).detach().numpy()
    prediction = logits.argmax(axis=-1)
    return CLASS_LABELS[prediction[0]], probabilities[0][prediction[0]]

# Function to perform NER
def extract_event_details(text):
    ner_pipeline = pipeline("ner", model=token_classifier, tokenizer=tokenizer, aggregation_strategy="simple")
    entities = ner_pipeline(text)
    event_details = {"description": [], "location": [], "date/time": []}
    for entity in entities:
        if entity['entity_group'] == 'B-DESC':
            event_details['description'].append(entity['word'])
        elif entity['entity_group'] == 'B-LOC':
            event_details['location'].append(entity['word'])
        elif entity['entity_group'] == 'B-DATE':
            event_details['date/time'].append(entity['word'])
    
    # Joining the words to form a complete text for each entity type
    for key, value in event_details.items():
        event_details[key] = ' '.join(value)

    return event_details

# Main function
def main(text):
    # Check if the text is about an event
    label, probability = classify_event(text)
    if label == "event":
        event_details = extract_event_details(text)
        return event_details
    else:
        return f"The text does not contain event information (probability: {probability:.2f})"

if __name__ == "__main__":
    text = "June 2022 15: Read-A-Thon Pizza Party for Winning Teams @3:00 pm 16: Open HSA Meeting @ 9:00 am 21-23: 1/2 Days of School 21: 8th Grade Promotion @ 6:00 pm 23: Last Day of School"
    result = main(text)
    print(result)
