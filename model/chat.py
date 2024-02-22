import random
import json
import os
import torch

from model import NeuralNet
from nltk_utils import bag_of_words
from pythainlp import word_tokenize
from pythainlp.spell import correct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

url = os.path.join('database/intents.json')
with open(url, 'r',encoding='utf-8') as  json_data:
    intents = json.load(json_data)


FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]



model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
def get_response(msg):
    
    sentence = word_tokenize(msg)
    ' '.join(sentence)
    print(sentence)
    
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "q":
            break

        resp = get_response(sentence)
        print(resp)
        
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "q":
#         break
    
#     #ต้องลง PyThaiNLP ก่อน เป็น library ที่ทำให้ NLP รองรับภาษาไทย
#     #pip install python-crfsuite
#     #pip install --upgrade --pre pythainlp
#     sentence = word_tokenize(sentence)
#     ' '.join(sentence)
#     print(sentence)
    
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.5:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")
