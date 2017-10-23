import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Bot(nn.Module):

    def __init__(self):
        super(Bot, self).__init__()

        # self.fc1 = nn.Linear(2, 10)
        # self.fc2 = nn.Linear(10, 1)

        self.cc1 = nn.Conv2d(3, 10, 7, stride=1, padding=3)
        self.cc2 = nn.Conv2d(10, 30, 7, stride=1, padding=3)
        self.cc3 = nn.Conv2d(30, 30, 5, stride=1, padding=2)
        self.cc4 = nn.Conv2d(30, 3, 5, stride=1, padding=2)
        # self.cc3 = nn.ConvTranspose2d(10, 5, 3)
        # self.cc4 = nn.ConvTranspose2d(5, 3, 3)


    def forward(self, x):

        x = F.tanh(self.cc1(x))
        x = F.tanh(self.cc2(x))
        x = F.tanh(self.cc3(x))
        x = F.tanh(self.cc4(x))

        return x

    def my_train(self, inputs, labels):

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters())

        self.cuda()

        for epoch in range(1000):  # loop over the dataset multiple times
            if epoch % 100 == 0:
                print(epoch)
            running_loss = 0.0
            for i, data in enumerate(inputs):
                # get the inputs


                # wrap them in Variable
                ins, outs = Variable(inputs.cuda()), Variable(labels.cuda())

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