import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        if torch.cuda.is_available():
            self.cuda()

        self.layer1 = nn.Conv2d(7, 16, kernel_size=7, stride=1, padding=3)
        self.layer2 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3)
        self.layer3 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3)
        self.layer4 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.sigmoid(self.layer1(x))
        out = F.sigmoid(self.layer2(out))
        out = F.sigmoid(self.layer3(out))
        out = F.sigmoid(self.layer4(out))
        return out

    def my_train(self, inputs, labels):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(self.parameters())

        if False and torch.cuda.is_available():
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