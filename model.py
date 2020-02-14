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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3, 30)
        self.fc2 = nn.Linear(30, 1)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    
    def fit(self, trainloader, validationloader, loss, optim, lr, epochs, val_per_batch, cuda):
        trainlosses = []
        testlosses = []

        criterion = loss()
        optimizer = optim(params=self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            trainloss = 0
            self.train()
            for batch, (data, target) in enumerate(trainloader):
                data = data.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)
                if cuda:
                    image, target = image.cuda(), target.cuda()
        
                optimizer.zero_grad()
        
                output = self.forward(data)
                loss = criterion(output, target)
        
                loss.backward()
                optimizer.step()
        
                trainloss += loss.item()
        
                if batch % val_per_batch == 0:
                    testloss = 0
                    self.eval()
                    with torch.no_grad():
                        for data, target in validationloader:
                            data = data.type(torch.FloatTensor)
                            target = target.type(torch.FloatTensor)
                            if cuda:
                                image, target = image.cuda(), target.cuda()
                            ps = self.forward(data)
                            testloss += criterion(ps, target).item()
                        testloss = testloss / len(validationloader)        
                    trainloss = trainloss / len(trainloader)
        
                trainlosses.append(trainloss)
                testlosses.append(testloss)
        
                print(f'Epoch: {epoch}',
                      f'Batch: {batch} out of {len(trainloader)}',
                      f'Training Loss: {trainloss}',
                      f'Test Loss: {testloss}')
        return (trainlosses, testlosses)
    
    def test(self, testloader):
        result = []
        expected = []
        for data, label in testloader:
            for res in self.forward(data.type(torch.FloatTensor)).tolist():
                result.append(res[0])
            for expt in label.tolist():
                expected.append(expt[0])
        return (result, expected)