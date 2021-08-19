import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import os

import matplotlib.pylab as plt
import matplotlib.image as img

costum_test = datasets.ImageFolder(root = 'C:\deeplearning_0toall_\costum set', transform = transforms.ToTensor())
                            
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed_all(1)

mnist_train = datasets.MNIST(root='MNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
mnist_test = datasets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

learning_rate = 0.01
training_epochs = 15
batch_size = 100

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, shuffle=True, drop_last=True, batch_size = batch_size)


class MnistNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.sq = nn.Sequential(
            nn.Linear(16*24*24, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )
        self.sq2 = nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.weightInitializer()
    
    def forward(self, x):
        out = self.sq2(x)
        out = out.view(-1,16*24*24)
        out = self.sq(out)
        return out
    
    def weightInitializer(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight)


model = MnistNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)

for epoch in range(training_epochs):

    avg_cost = 0
    
    for X, Y in data_loader:
        X = X.to(device)# Before X.shape = torch.Size([100, 1, 28, 28]) After torch.Size([100, 784])
        Y = Y.to(device)

        pred = model(X)

        cost = F.cross_entropy(pred, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    
    correct_prediction = (torch.argmax(pred, 1) == Y)
    accuracy = correct_prediction.float().mean()
    print('Epoch: {:d}/15, Cost: {:.6f}, Acc: {:.6f}'.format(epoch+1, avg_cost,accuracy))

with torch.no_grad(): 
    X_test = mnist_test.data.view(-1,1,28,28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    pred = model(X_test)
    correct_prediction = (torch.argmax(pred, 1) == Y_test)
    accuracy = correct_prediction.float().mean()

    print('Accuracy: {}'.format(accuracy))

tf = transforms.ToTensor()

with torch.no_grad():
    for i in os.listdir('C:/deeplearning_0toall_/costum set/abs'):
        p = 'C:/deeplearning_0toall_/costum set/abs/' + str(i)
        aaaa = img.imread(p)
        bbbb = torch.abs(1 - tf(aaaa))
        X_test = bbbb[0,:,:].view(-1,1,28,28).float().to(device)
        pred = model(X_test)
        print('Prediction: ', torch.argmax(pred).item())
        plt.imshow(bbbb[0,:,:], cmap='Greys', interpolation='nearest')
        plt.show()
# with torch.no_grad():
#     X_test = mnist_test.test_data.view(-1,1,28,28).float().to(device)
#     Y_test = mnist_test.test_labels.to(device)

#     prediction = model(X_test)
#     correct_prediction = torch.argmax(prediction, 1) == Y_test
#     accuracy = correct_prediction.float().mean()

#     print('Accuracy:', accuracy.item())

#     #Get one and Predict
#     r = random.randint(0, len(mnist_test) - 1)
#     X_single_data = mnist_test.test_data[r: r+1].view(-1,1,28,28).float().to(device)
#     Y_single_data = mnist_test.test_labels[r: r+1].to(device)

#     print('Label: ', Y_single_data.item())
#     single_prediction = model(X_single_data)
#     print('Prediction: ', torch.argmax(single_prediction).item())

#     plt.imshow(mnist_test.test_data[r: r+1].view(28, 28), cmap='Greys', interpolation='nearest')
#     plt.show()