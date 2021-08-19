#주석이 코드의 위에 있습니다. 유의하시고 읽어주세요.
#박진성이 웹에서 짜집기 함, 최수형선배님께서 전반적으로 봐주심.

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
 
# cuda를 적용 시키는 코드입니다. GPU연산을 통해서 속도를 비약적으로 향상시킬수 있습니다.
#cmd에 아래 코드를 쳐주면 cuda 및 pytorch가 호환되는 버전으로 깔립니다. (이미 설치되어있는 경우에도 사용 가능합니다.)
#pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed_all(1)

#데이터셋을 지정하는 코드들입니다. 딥러닝에 있어 데이터셋 지정등은 부가적인 기능이므로 만약 학습에 익숙하지 않다면 지금은 생략해도 좋습니다.
mnist_train = datasets.MNIST(root='MNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
mnist_test = datasets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

#인간이 정해주는 인수인 하이퍼파라미터 조정부입니다.
#학습률입니다.
learning_rate = 0.01
#몇번 학습할껀지 정합니다
training_epochs = 15
#한번에 몇개 학습할껀지 정합니다.
batch_size = 100

#학습 데이터를 로드합니다. batch사이즈에 맞게 데이터를 로드하는 역할입니다.
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, shuffle=True, drop_last=True, batch_size = batch_size)

#신경망을 클래스로써 정의합니다. 이렇게 클래스로 정의하면 다양한 기능들을 모듈식으로 쉽게 가져올수 있고, 관리및 확장에 용이합니다.
class MnistNN(nn.Module):
    #처음 신경망이 생성되었을때 실행되는 코드입니다.
    def __init__(self):
        #nn.Module을 상속하여 각종 설정들을 자동 적용해줍니다.
        #이후에 여러 신경망을 모듈식으로 가져올수 있게 해줍니다.
        super().__init__()
        #nn.Sequential으로 연속되는 신경망을 만듭니다.
        #입력층의 노드는 28*28개, 첫번째 은닉층의 노드는 2048개,
        #두번째 은닉층의 노드도 2048개, 출력층의 노드는 10개입니다.
        #활성화함수로는 ReLU함수를 이용하였습니다.
        self.sq = nn.Sequential(
            nn.Linear(28*28, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )
        #신경망 구성이 끝난 후에는 가중치 초기화를 해줍니다. 일반적으로 훨씬 더 빠르게 학습이 될수 있도록 도와줍니다.
        self.weightInitializer()
    
    #순전파 함수를 정의합니다. 기본적으로 model(x)를 하면 foward가 실행됩니다.
    #순전파란 신경망에 데이터를 넣어, 출력값을 받는겁니다.
    def forward(self, x):
        out = self.sq(x)
        return out
    
    #가중치를 초기화해줍니다. ReLU등은 가중치가 없으므로 생략하고,
    #레이어타입(리니어, convolution2d, Batchnorm)등등에 대해 다른 초기값을 지정할수 있도록 코드를 적습니다.
    def weightInitializer(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight)

#위에서 선언한 신경망 클래스로, model 객체를 만듭니다. 뒤에 붙은 .to(device)는 cuda를 연결해서 연산할수 있게 해줍니다.
model = MnistNN().to(device)

#optimizer를 설정합니다. 대부분의 상황에 대해 Adam이 성능상 우세함으로 그냥 Adam을 쓰도록 했습니다.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(data_loader)

#학습을 epoch번 실행합니다.
for epoch in range(training_epochs):

    #결과 표시를 위한 코드의 일부입니다.
    avg_cost = 0
    
    #데이터 로더에서 데이터를 꺼내서 신경망에 넣습니다.
    for X, Y in data_loader:
        #신경망의 첫 레이어는 (28*28)사이즈이므로 .view()으로 텐서의 모양을 조정해줍니다.
        X = X.view(-1,28*28).to(device)
        Y = Y.to(device)

        #예측값을 model의 순전파값으로 정의합니다.
        pred = model(X)

        #cost function입니다. classification이므로 cross_entropy를 사용하였습니다.
        #위의 신경망에 마지막에 소프트맥스 함수가 없는 이유는 cost function에서 cross_entropy를 사용하면 자동으로 소프트맥스가 적용되기 때문입니다.
        cost = F.cross_entropy(pred, Y)

        #optimizer에 들어있는 기존 누적값을 초기화합니다.
        optimizer.zero_grad()
        #신경망에 대해 역전파를 합니다.
        cost.backward()
        #optimizer로 가중치를 업데이트 합니다.
        optimizer.step()

        #결과 표시를 위한 코드의 일부입니다.
        avg_cost += cost / total_batch
    
    #결과 표시를 위한 코드의 일부입니다.
    correct_prediction = (torch.argmax(pred, 1) == Y)
    accuracy = correct_prediction.float().mean()
    print('Epoch: {:d}/15, Cost: {:.6f}, Acc: {:.6f}'.format(epoch+1, avg_cost,accuracy))

#테스트 시작입니다. no_grad()는 가중치 업데이트를 안하겠다는 뜻입니다.
# 테스트를 함으로 도중에 업데이트가 되어서는 않되기에 사용합니다.
with torch.no_grad(): 
    #테스트 데이터셋을 가져와줍니다. data_loader로 batch사이즈많큼 떼어서 가져올 필요가 없으니 데이터셋에서 바로 연결해줍니다.
    #신경망의 첫 레이어는 (28*28)사이즈이므로 .view()으로 텐서의 모양을 조정해줍니다.
    X_test = mnist_test.data.view(-1,28*28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    #예측값을 model의 순전파값으로 정의합니다.
    pred = model(X_test)

    #결과 표시를 위한 코드의 일부입니다.
    correct_prediction = (torch.argmax(pred, 1) == Y_test)
    accuracy = correct_prediction.float().mean()

    print('Accuracy: {}'.format(accuracy))

# 커스텀 이미지로 테스트 할수 있는 코드

# costum_test = datasets.ImageFolder(root = 'C:\deeplearning_0toall_\costum set', transform = transforms.ToTensor()) #경로를 지정해주세요
# tf = transforms.ToTensor()
# with torch.no_grad(): 
#     for i in os.listdir('C:/deeplearning_0toall_/costum set/abs'): #경로를 지정해주세요
#         p = 'C:/deeplearning_0toall_/costum set/abs/' + str(i) #경로를 지정해주세요
#         aaaa = img.imread(p)
#         bbbb = torch.abs(1 - tf(aaaa))
#         X_test = bbbb[0,:,:].view(-1,28*28).float().to(device)
#         pred = model(X_test)
#         print('Prediction: ', torch.argmax(pred).item())
#         plt.imshow(bbbb[0,:,:], cmap='Greys', interpolation='nearest')
#         plt.show()
