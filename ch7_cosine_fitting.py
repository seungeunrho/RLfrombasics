import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

data_x = np.random.rand(10000) * 5
model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(2000):
    batch_x = np.random.choice(data_x, 32) # 랜덤하게 뽑힌 32개의 데이터로 mini-batch를 구성
    batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1)
    pred = model(batch_x_tensor)

    batch_y = true_fun(batch_x)
    truth = torch.from_numpy(batch_y).float().unsqueeze(1)
    loss = F.mse_loss(pred, truth) #로스 함수를 계산하는 부분
    
    optimizer.zero_grad() 
    loss.mean().backward() # back propagation을 통한 편미분 계산이 실제로 일어나는 부분
    optimizer.step() # 파라미터를 업데이트 하는 부분
        
import matplotlib.pyplot as plt
x = np.linspace(0, 5, 100)
input_x = torch.from_numpy(x).float().unsqueeze(1)
plt.plot(x, true_fun(x), label="Truth")
plt.plot(x, model(input_x).detach().numpy(), label="Prediction")
plt.legend(loc='lower left',fontsize=15)
plt.xlim((0, 5))
plt.ylim((-2, 1.5))
plt.grid()
