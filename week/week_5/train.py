import os #
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

lr = 0.001
image_size = 28
num_classes = 10
batch_size = 100
hidden_size = 500
total_epochs = 3
results_folder = 'results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
if not os.path.exist(results_folder):
    os.makedirs(results_folder)
    
folder_name = max([0] + [int(e) for e in os.listdir(results_folder)])+1
save_path = os.path.join(results_folder, str(folder_name))
os.makedirs(save_path)
#

with open(os.path.join(save_path, 'hparam.txt'), 'w') as f:
    f.write(f'{lr}\n')
    f.write(f'{image_size}\n')
    f.write(f'{num_classes}\n')
    f.write(f'{batch_size}\n')
    f.write(f'{hidden_size}\n')
    f.write(f'{total_epochs}\n')
    f.write(f'{results_folder}\n')
    
class MLP(nn.Module) :
    def __init__(self, image_size, hidden_size, num_classes) :
        super().__init__()
        
        self.image_size = image_size
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
    def forward(self, x) :
        batch_size = x.shape[0]
        
        x = torch.reshape(x, (-1,self.image_size*self.image_size))
        
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        
        return x

    
myMLP = MLP(image_size, hidden_size, num_classes).to(device)
    
train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)      

train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

optim = Adam(params=myMLP.parameters(), lr=lr)

def evaluate(model, testloader, device):
    model.eval()
    total = 0
    correct = 0
    
    for image, label in testloader:
        image, label = image.to(device), label.to(device)
        output = model(image)
        output_index = torch.argmax(output, dim=1)
        
        correct += (output_index == label).sum().item()
        total += label.shape[0]
    acc = correct / total * 100
    model.train()
    return acc

def evaluate_by_class(model, testloader, device, num_classes):
    model.eval()
    total = torch.zeros(num_classes)
    correct = torch.zeros(num_classes)
    
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        output = model(image)
        output_index = torch.argmax(output, dim=1)
        
        for idx in range(num_classes):
            total[idx] += (label == idx).sum().item()
            correct[idx] += ((label == idx) * (output_index == idx)).sum(). item()
            
    acc = correct / total
    model.train()
    return acc

max = -1
    

for epoch in range(total_epochs) :
    
    for idx, (image, label) in enumerate(train_loader) :
        image = image.to(device)
        label = label.to(device)
        
        output = myMLP(image)
        
        loss = loss_fn(output, label)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        if idx % 100 == 0:
            print(loss)
            
            acc = evaluate(myMLP, test_loader, device)
            
            if max < acc:
                print("새로운 max 값 달성. 모델 저장", acc)
                max = acc
                
                torch.save(
                    myMLP.state_dict(),
                    os.path.join(save_path, 'myMLP_best.ckpt')
                )
            


