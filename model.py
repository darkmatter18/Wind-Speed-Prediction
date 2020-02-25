#    Copyright 2020 Arkadip Bhattacharya

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Progbar

class Model(nn.Module):
    def __init__(self,input_size = 3, lstm_input_size = 51, lstm_hidden_size = 102, time_series = 6, lstm_num_layers = 1, cuda=False):
        super(Model, self).__init__()
        
        lstm_drop = 0.2 if lstm_num_layers > 1 else 0 
        
        self.lstm_num_layers = lstm_num_layers
        self.cuda = cuda
        self.hidden_size = lstm_hidden_size
        
        # Model Architucture Starts
        self.conv1 = nn.Conv1d(time_series, 18, 1)
        self.pool1 = nn.MaxPool1d(1)
        self.drop1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(18 * input_size, lstm_input_size)
        self.drop2 = nn.Dropout(p=0.2)
        
        self.lstm3 = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first= True,
                            dropout = lstm_drop)
        self.drop3 = nn.Dropout(p=0.2)
        
        self.fc4 = nn.Linear(lstm_hidden_size, lstm_input_size)
        self.drop4 = nn.Dropout(p=0.2)
        
        self.fc5 = nn.Linear(lstm_input_size, time_series)
        

    def init_hidden(self, batch_size):
        device = torch.device('cuda') if self.cuda else torch.device('cpu')
        return (torch.zeros(self.lstm_num_layers, batch_size, self.hidden_size, device = device),
                torch.zeros(self.lstm_num_layers, batch_size, self.hidden_size, device = device))
        
    def forward(self, x):
        self.hidden = self.init_hidden(x.shape[0])
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        
        x = x.unsqueeze(1)
        
        x, self.hidden = self.lstm3(x, self.hidden)
        x = self.drop3(x)
        
        x = self.drop4(F.relu(self.fc4(x)))
        
        x = self.fc5(x)
        
        return x
    
    
    def fit(self, trainloader, validationloader, loss, optim, lr, epochs, val_per_batch):
        trainlosses = []
        testlosses = []

        self.criterion = loss()
        self.optimizer = optim(params=self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            trainloss = 0
            progbar = Progbar(target=len(trainloader) - 1)
            for batch, (data, target) in enumerate(trainloader):
                self.train()
                data = data.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
        
                self.optimizer.zero_grad()
        
                output = self.forward(data)
                # loss = self.criterion(output.view(-1, 1), target.view(-1))
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
        
                trainloss += loss.item()
        
                if batch % val_per_batch == 0:
                    testloss = 0
                    self.eval()
                    with torch.no_grad():
                        for data, target in validationloader:
                            data = data.type(torch.FloatTensor)
                            target = target.type(torch.FloatTensor)
                            if self.cuda:
                                data, target = data.cuda(), target.cuda()
                            ps = self.forward(data)
                            testloss += self.criterion(ps, target).item()
                        testloss = testloss / len(validationloader)        
                    trainloss = trainloss / len(trainloader)
        
                trainlosses.append(trainloss)
                testlosses.append(testloss)
                
                progbar.update(current=batch, values=[('Epoch', epoch+1), ('Training Loss', trainloss), ('Test Loss', testloss)])
#                 print(f'Epoch: {epoch+1}',
#                       f'Batch: {batch} out of {len(trainloader)}',
#                       f'Training Loss: {trainloss}',
#                       f'Test Loss: {testloss}')
        self.trainlosses = trainlosses
        self.testlosses = testlosses
        return (trainlosses, testlosses)
    
    def test(self, testloader):
        result = []
        expected = []
        self.eval()
        with torch.no_grad():
            for data, label in testloader:
                data = data.type(torch.FloatTensor)
                if self.cuda:
                    data, label = data.cuda(), label.cuda()
                for res in self.forward(data).cpu().numpy().flatten():
                    result.append(res.item())
                for expt in label.cpu().numpy().flatten():
                    expected.append(expt.item())
        
        self.result = result
        self.expected = expected
        return (result, expected)
    
    def save(self, path, save_optim=False):
        try:
            os.mkdir(os.path.join('model', path))
            print(path, " - dir Created")
        except FileExistsError:
            print(path, " - dir Already exists")
        finally:
            NAME = 'model_cuda.pt' if self.cuda else 'model_cpu.pt'
            torch.save(self.state_dict(), os.path.join('model', path, NAME))
            print('Model saved in', path)
            if save_optim:
                NAME = 'optim_cuda.pt' if self.cuda else 'optim_cpu.pt'
                torch.save(self.optimizer.state_dict(), os.path.join('model', path, NAME))
                print('Optimizer saved in', path)
        return