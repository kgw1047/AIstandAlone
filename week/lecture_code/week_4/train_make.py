# import
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

# 모델을 정의하기 위한 하이퍼파라미터 선언, 학습에 필요한 값도
# 학습률 선언
lr = 0.001
# 입력 받을 이미지의 크기 선언
image_size = 28
# 출력(클래스)의 수 선언
num_classes = 10
# 배치 크기 선언
batch_size = 100
# 은닉층의 노드 수 선언
hidden_size = 500
# 학습을 반복할 횟수 선언
epoch = 3

# 데이터를 보낼 디바이스를 가능하면 gpu, 아니면 cpu로 선언
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델의 설계도 작성, 우리가 만들 모델이 어떤 구조를 가지고 있는지
# 모델의 설계도 만들기, 기본적으로 딥러닝 모델들은 pytorch의 nn.Module이라는 객체를 상속
class MLP(nn.Module) :    
# 객체의 init함수 선언, 객체가 가질 요소들을 선언하는 곳, 이 때 외부의 값이 필요하다면 파라미터로 받음
    def __init__(self, image_size, hidden_size, num_classes) :
# 해당 객체가 상속 받는 객체(nn.Module)의 init함수에 존재하는 요소들을 선언
        super().__init__()
# 입력 받을 이미지의 크기를 나타내는 변수 선언, 
        self.image_size = image_size
# 단순히 하이퍼파라미터를 선언할 때의 image_size를 사용하지 않는 것은 해당 파일이 여러 파이썬 파일로 나누어질 때를 대비

# 예제로 주어진 모델의 첫 번째 레이어를 선언, 파라미터 부분에서 실제값을 사용하는 것이 아닌 외부에 선언된 하이퍼파라미터를 받는 형태
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
# 예제로 주어진 모델의 두 번째 레이어를 선언
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
# 예제로 주어진 모델의 세 번째 레이어를 선언
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
# 예제로 주어진 모델의 마지막 레이어를 선언, 최종적으로 우리가 원하는 출력의 크기(클래스)를 출력
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
# 모델이 예측값을 출력하기 위한 필수적인 forward 함수
# 딥러닝 모델에서 필수로 선언해야하는 forward 함수 선언, self는 필수, 모델이 에측을 진행하는 과정이므로 최초의 입력 x가 필요
    def forward(self, x) :
# x = [batch_size, 28, 28, 1] 의 크기를 가지고 입력, 
        batch_size = x.shape[0]
         # torch.reshape(input, shape) => 입력 x를 모델에 들어가기에 적절한 사이즈로 조절
         # 28*28의 행을 가진 형태의 행렬로 입력을 받는다.
        x = torch.reshape(x, (-1, self.image_size*self.image_size))
# 입력을 모델의 레이어에 차례로 통과시킨다.
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        
# 모든 레이어를 통과한 최종 출력값을 반환
        return x

# 앞서 만들어놓은 설계도를 통해 모델을 생성(하이퍼파라미터 이용), 해당 모델을 위에서 결정된 device로 전달
myMLP = MLP(image_size, hidden_size, num_classes).to(device)
    
# Dataset 선언, 해당 학습 방식에서는 train과 test가 존재
# Train Dataset(알바생) 선언, MNIST 데이터셋을 다운받아 Tensor로 변환한 후 root에 지정된 경로에 저장, 정규화만 진행하는 transform을 선택
train_set = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
# Test Dataset(알바생) 선언, MNIST 데이터셋을 다운받아 Tensor로 변환한 후 root에 지정된 경로에 저장, 정규화만 진행하는 transform을 선택
test_set = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)

# DataLoader 선언, 해당 학습 방식에서는 train과 test가 존재
# Train Dataloader(사서) 선언, 알바생(Dataset)과 쌓이길 기다리는 책의 양(batch_size), shuffle 여부를 같이 선언
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# Test Dataloader(사서) 선언, 알바생(Dataset)과 쌓이길 기다리는 책의 양(batch_size), shuffle 여부를 같이 선언
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Loss 함수 선언, torch.nn에 존재하는 CrossEntropyLoss 객체를 활용
loss_fn = nn.CrossEntropyLoss()
# Optimizer 선언, Adam optimizer를 활용, params는 optimizer가 업데이트할 가중치를 지정. 해당 모델은 존재하는 모든 가중치를 업데이트
optim = Adam(params=myMLP.parameters(), lr=lr)
# 본격적인 학습을 진행하는 부분
# 모든 데이터를 통해 가중치를 한 번 업데이트하면 1 epoch, 하이퍼파라미터로 지정한 epoch만큼 반복
for epoch in range(epoch) :
# batch_size가 100이므로 한 번에 100개의 이미지를 모델로 전달(100, 28*28)의 형태, 이를 총 600번 실행, (100, 28*28)의 요소가 600개 존재하는 list라고 생각하면 좀 편함.
    for idx, (image, label) in enumerate(train_loader) :
# 입력 받은 이미지를 device로
        image = image.to(device)
# 입력 받은 라벨을 device로
        label = label.to(device)
        
# 출력값(예측값)을 output으로 저장
        output = myMLP(image)
# 앞서 만든 loss_fn을 통해 loss값을 계산
        loss = loss_fn(output, label)
# 모델이 예측을 진행한 역방향으로 미분값 계산
        loss.backward()
# optimizer를 통해 1번 업데이트 진행
        optim.step()
# 기본적으로 미분값들을 저장하는데, 해당 모델은 사용하지 않으므로 삭제
        optim.zero_grad()
        
# 학습 중간에 loss를 확인하기 위한 출력문
        if (idx % 100 == 0) :
            print(loss)



