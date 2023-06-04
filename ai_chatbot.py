"""
AI Chatbot

This script implements a simple chatbot using a neural network model.
The chatbot is trained on a set of predefined patterns and intents stored in a
JSON file. It uses a bag-of-words representation of user input to predict the
corresponding intent and generate appropriate responses.

Author: ZION


The script performs the following steps:

    Load the training data from a JSON file containing patterns, intents, and responses.
    Preprocess the training data by tokenizing, stemming, and creating bags of words.
    Configure and train a neural network model using TensorFlow and TFLearn libraries.
    Handle user input by converting it into a bag-of-words representation and predicting the intent using the trained model.
    Generate appropriate responses based on the predicted intent.
    Provide the option to add new patterns and responses to the training data.

Functions:
    bag_of_words()  Convert user input into a bag-of-words representation for the neural network model.
    chat()          Handle the conversation with the user, predict intents, and generate responses.
    add_to_json()   Allow the user to add new patterns and responses to expand the chatbot's knowledge.

Usage:
    Ensure the "intents.json" file contains the training data in the appropriate format.
    Run the script and start interacting with the chatbot.

Note:
    The script requires the nltk, numpy, tensorflow.compat.v1, and tflearn libraries to be installed.
"""

import json
import pickle
from os import environ, path
from random import choice
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from numpy import argmax, array
from tensorflow.compat.v1 import reset_default_graph
from tflearn import DNN, embedding, fully_connected, input_data, lstm, regression

stemmer = LancasterStemmer()

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

with open("intents.json", encoding="utf-8") as data_file:
    data = json.load(data_file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = array(training)
    output = array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

reset_default_graph()

net = input_data(shape=[None, len(training[0])])
net = embedding(net, input_dim=len(words), output_dim=128)
net = lstm(net, 128, dropout=0.8)
net = fully_connected(net, len(output[0]), activation="softmax")
net = regression(net)

model = DNN(net, tensorboard_dir="vegapunk1_tflearn_logs")

if path.exists("model.tflearn.index"):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(inp, new_words):
    new_bag = [0 for _ in range(len(new_words))]

    s_words = word_tokenize(inp)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for word in s_words:
        for i, j in enumerate(new_words):
            if j == word:
                new_bag[i] = 1

    return array(new_bag)


def chat():
    print("Bot: Hello! How can I assist you?")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        elif inp.lower() == "add":
            add_to_json()
            continue

        results = model.predict([bag_of_words(inp, words)])
        results_index = argmax(results)
        tag = labels[results_index]

        for new_tag in data["intents"]:
            if new_tag["tag"] == tag:
                responses = new_tag["responses"]

        print("Bot:", choice(responses))


def add_to_json():
    print("Bot: Sure! Let's add new information to the bot's knowledge.")
    tag = input("Enter the tag: ")
    patterns = []
    responses = []

    print("Enter patterns (or 'done' to finish):")
    while True:
        pattern = input("> ")
        if pattern.lower() == "done":
            break
        patterns.append(pattern)

    print("Enter responses (or 'done' to finish):")
    while True:
        response = input("> ")
        if response.lower() == "done":
            break
        responses.append(response)

    new_intent = {
        "tag": tag,
        "patterns": patterns,
        "responses": responses,
        "context_set": "",
    }

    existing_tags = [intent["tag"] for intent in data["intents"]]
    if tag in existing_tags:
        print("Bot: The tag already exists. Updating the information...")
        for intent in data["intents"]:
            if intent["tag"] == tag:
                intent["patterns"] = patterns
                intent["responses"] = responses
                break
    else:
        print("Bot: Adding new information...")
        data["intents"].append(new_intent)

    with open("intents.json", "w") as file:
        json.dump(data, file)

    print("Bot: New information added to intents.json.")


chat()
