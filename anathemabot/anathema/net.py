import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(7, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc(out)
        return out

    def my_train(self, inputs, labels):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(self.parameters())

        if torch.cuda.is_available():
            self.cuda()

        for epoch in range(1000):  # loop over the dataset multiple times
            if epoch % 100 == 0:
                print(epoch)
            running_loss = 0.0
            for i, data in enumerate(inputs):
                # get the inputs


                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                ins, outs = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(ins)
                loss = criterion(outputs, outs)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0