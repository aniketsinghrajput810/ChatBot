from django.shortcuts import render
from django.http import JsonResponse
import nltk
import numpy as np
import random
import string
import bs4 as bs
import urllib.request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Fetching data from Wikipedia about Python programming language
link = urllib.request.urlopen('https://en.wikipedia.org/wiki/Python_(programming_language)')
link = link.read()
data = bs.BeautifulSoup(link, 'lxml')
data_paragraphs = data.find_all('p')
data_text = ''
for para in data_paragraphs:
    data_text += para.text

data_text = data_text.lower()
data_text = re.sub(r'\[[0-9]*\]', ' ', data_text)
data_text = re.sub(r'\s+', ' ', data_text)

# Tokenization of text
sen = nltk.sent_tokenize(data_text)
words = nltk.word_tokenize(data_text)

# Lemmatization of words
wnlem = nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return [wnlem.lemmatize(token) for token in tokens]

# Define function to preprocess text
pr = dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(pr)))

# Define inputs and responses for greetings
greeting_inputs = ("hey", "hello", "good morning","how are you", "good evening", "morning", "hi", "what's up?")
greeting_responses = ["hey", "hey, how are you?", "hello, how are you doing?", "hello", "welcome", "welcome. I am good, how about you?"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)

# Define function to generate bot responses
def generate_response(user_input):
    bot_response = ''
    sen.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    word_vectors = word_vectorizer.fit_transform(sen)
    similar_vector_values = cosine_similarity(word_vectors[-1], word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "I am sorry I don't understand"
    else:
        bot_response = bot_response + sen[similar_sentence_number]

    sen.pop()
    return bot_response

# Define Django view function for chatbot
def chatbot(request):
    if request.method == "POST":
        user_input = request.POST.get('message')
        if user_input:
            user_input = user_input.lower()
            if user_input in ['bye', 'exit']:
                return JsonResponse({"response": "Good Bye"})
            greeting = generate_greeting_response(user_input)
            if greeting:
                return JsonResponse({"response": greeting})
            else:
                return JsonResponse({"response": generate_response(user_input)})
        else:
            return JsonResponse({"response": "I am sorry I don't understand"})
    return render(request, 'chatbot/chat.html')
