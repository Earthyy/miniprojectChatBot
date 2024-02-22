import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    # Model รับค่า words เข้ามา traing ในรูปเเบบของเมทริกซ์
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size) 
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.l2 = nn.Linear(hidden_size, hidden_size) 
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.l3 = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         out = self.l1(x)
#         out = torch.relu(out)
#         out = self.dropout1(out)
#         out = self.l2(out)
#         out = torch.relu(out)
#         out = self.dropout2(out)
#         out = self.l3(out)
#         # no activation and no softmax at the end
#         return out
