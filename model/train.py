import numpy as np
import random
import json
import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

#ทำให้รองรับภาษาไทย
url = os.path.join('database/intents.json')
with open(url, 'r',encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Loop ค่าข้อมูลในไฟล์ Json
for intent in intents['intents']:
    tag = intent['tag']
    # เพิ่มข้อมูลลงในส่วนท้ายของลิสต์
    tags.append(tag)
    for pattern in intent['patterns']:
        # call tokenize 
        w = tokenize(pattern)
        # รวม List เข้าด้วยกัน
        all_words.extend(w)
        # เพิ่มข็อมูลลงในส่วนท้ายของ Liset ทำให้เกิด list 2 ตัว ใน List เดียว
        xy.append((w, tag))

# call stem เเละตัดเครื่องหมายที่ไม่ต้องการทิ้ง
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# เรียงตัวข้อมูล
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy),xy, "\n")
print(len(all_words),all_words, "\n")
print(len(tags),tags, "\n")

X_train = []
y_train = []
# Loop ข้อมูลที่อยู่ใน List xy
for (pattern_sentence, tag) in xy:
    # call function BoW 
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # สร้างป้ายกำกับเพื่อนำไปใช้ใน CrossEntropyLoss เเละส่วนอื่นๆ
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
print("\n",X_train,"\n")
print("\n",y_train,"\n")


# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

#ใช้สำหรับเเยกชุดข้อมูลออกจากโมเดลเพราะว่าเวลาฝึกโมเดลจะมีการเปรับเปลี่ยนค่าของชุดข้อมูล
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # เข้าถึงตำเเหน่งข้อมูลใน X_train และ y_train ด้วย index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # เมื่อมีการเรียกใช้ class จะรีเทรนขนาดของ  dataset
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  
#num_workers=0 เราจะให้ Pytorch เลือกทรัพยากรอัตโนมัติ เพราะเราจะทำงานบน GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass นำข้อมูลเข้า input layer ใน NN
        #words คือ x_data ที่อยู่ใน class ChatDataset 
        outputs = model(words) #words เป็นข้อมูลของ pattren ทั้งหมดที่ทำการเเปลงค่าเป็น 0 กับ 1 แล้ว
        # print("\n",outputs)
        
        #Label คือ y_data ที่อยู่ใน class ChatDataset
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad() #set ค่า optimizer parameters เป็น 0
        loss.backward() #ใช้ Backpropagation กับค่าของพารามิเตอร์ของ Loss 
        #new weight = weight - (learning rate * loss)
        optimizer.step() # update ค่าพารามิเตอร์ของ optimizer  
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', outputs)


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
