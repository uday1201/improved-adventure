import os
from email.parser import BytesParser
from email.policy import default
from email import message_from_bytes
from bs4 import BeautifulSoup
import re

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

def extract_event_details(email_text):
    events = []

    # Split email text into lines
    lines = email_text.split('\n')

    # Search for dates and events in each line
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse a date in the line
        try:
            date = parse(line, fuzzy=True)
            event = {"date": date.strftime("%Y-%m-%d")}
        except ValueError:
            event = {}

        # Search for event names and locations
        match = re.search(r'(\b[A-Za-z]+[\sA-Za-z]*\b)(?:.*(?:at|on|in|during|through)\s)?(\b[A-Za-z\s\d.,]+\b)?', line)
        if match:
            event_name = match.group(1).strip()
            location = match.group(2)
            if event_name and not event_name.isdigit():
                event["event_name"] = event_name
            if location and not location.isdigit():
                event["location"] = location.strip()

        if "event_name" in event:
            events.append(event)

    return events

import spacy
from dateparser import parse

nlp = spacy.load("en_core_web_sm")

def extract_events_unstructured(email_body):
    events = []
    doc = nlp(email_body)

    for ent in doc.ents:
        if ent.label_ == "DATE" or ent.label_ == "TIME":
            event_date = parse(ent.text)
            if event_date:
                # Find a noun or proper noun closest to the date entity that could be the event title
                event_title = None
                for token in ent:
                    if token.dep_ == "compound" or token.dep_ == "nsubj" or token.dep_ == "conj":
                        if token.head.pos_ == "NOUN" or token.head.pos_ == "PROPN":
                            event_title = token.head.text
                            break

                event = {
                    'title': event_title,
                    'date': event_date.isoformat()
                }
                events.append(event)

    return events

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

for line in emails:
    print(line)

# Example usage
# email_body = """
# There will be a Kids Art Workshop on April 12, 2023, at the Community Center.
# A fun and interactive workshop for kids aged 5-12.
# Don't miss the Family Fun Day on April 18, 2023, at the City Park.
# Enjoy a day of games, food, and entertainment for the whole family.
# """

# events = extract_events_unstructured(email_body)
# print(events)