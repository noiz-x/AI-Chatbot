# Simple Chatbot

This repository contains the code for a simple chatbot implemented using a neural network. The chatbot is trained to understand user input and generate appropriate responses based on predefined patterns and intents.

## Requirements

- Python (3.6 or higher)
- TensorFlow (1.15 or higher)
- TFLearn (0.3.2 or higher)
- NLTK (3.5 or higher)
- NumPy (1.19.5 or higher)
- JSON (built-in module)

## Installation

1. Clone the repository or download the code files.

2. Install the required dependencies using pip:

   ```
   pip install tensorflow tflearn nltk numpy
   ```

3. Download the NLTK data by running the following code in Python:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('lancaster')
   ```

4. Place your training data in a JSON file named "intents.json" following the provided format:

   ```json
   {
     "intents": [
       {
         "tag": "greeting",
         "patterns": ["Hi", "Hello", "Hey"],
         "responses": ["Hello! How can I assist you?"]
       },
       {
         "tag": "goodbye",
         "patterns": ["Bye", "See you later"],
         "responses": ["Goodbye! Have a great day!"]
       }
     ]
   }
   ```

   The file should contain a list of intents, where each intent has a unique tag, a list of patterns (user inputs), and a list of corresponding responses.

## Usage

1. Run the script by executing the following command:

   ```
   python chatbot.py
   ```

2. The chatbot will greet you and await your input:

   ```
   Bot: Hello! How can I assist you?
   You:
   ```

3. Enter your message or question and press Enter. The chatbot will generate a response based on the trained model:

   ```
   Bot: Hello! How can I assist you?
   You: Hi
   Bot: Hello! How can I assist you?
   ```

4. You can continue the conversation with the chatbot by entering additional messages. To exit the chatbot, simply type "quit".

5. To add new patterns and responses to the training data, type "add" when prompted. Follow the instructions to provide the new tag, patterns, and responses.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This chatbot implementation is based on the tutorial from [Python Engineer](https://www.youtube.com/watch?v=1lwddP0KUEg).
