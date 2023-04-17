import os
from email.parser import BytesParser
from email.policy import default
from email import message_from_bytes
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy

def read_email_body(file_path):
    with open(file_path, 'rb') as f:
        raw_email = f.read()

    # Parse the email from the raw bytes
    parsed_email = message_from_bytes(raw_email, policy=default)

    # Extract and return the email body
    email_body = ""
    if parsed_email.is_multipart():
        for part in parsed_email.walk():
            content_type = part.get_content_type()
            content_disposition = part.get("Content-Disposition", None)

            if content_type == "text/plain" and not content_disposition:
                email_body = part.get_payload(decode=True).decode()
                break
    else:
        email_body = parsed_email.get_payload(decode=True).decode()

    return email_body

emails = []

# Read samples
eml_file_path = "samples/"
for filename in os.listdir(eml_file_path):
    if filename.endswith('.eml'):
        file_path = os.path.join(eml_file_path, filename)
        email_body = read_email_body(file_path)
        # Remove HTML tags
        soup = BeautifulSoup(email_body, 'html.parser')
        clean_text = soup.get_text()
        # Remove extra whitespaces and line breaks
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Split the text into an array of lines without line breaks
        #lines = clean_text.split('. ')
        emails.append(clean_text)


# Clean up the email body
email_body = "June 2022 15: Read-A-Thon Pizza Party for Winning Teams @3:00 pm 16: Open HSA Meeting @ 9:00 am 21-23: 1/2 Days of School 21: 8th Grade Promotion @ 6:00 pm 23: Last Day of School"
print(email_body)

# Load the BERT model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")



# Create the NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Extract entities using the BERT model
entities = ner_pipeline(email_body)

# Load spaCy's English model for date/time recognition
nlp = spacy.load("en_core_web_sm")

# Extract date/time entities using spaCy
doc = nlp(email_body)
date_time_entities = [ent for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]

# Print the extracted event details
print("Event Details:")
for entity in entities:
    if entity['entity_group'] == "LOC":
        print("Location: ", entity["word"])
for date_time_entity in date_time_entities:
    print("Date/Time: ", date_time_entity.text)
