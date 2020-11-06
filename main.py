import torch
import torch.nn as nn	# 各种层类型的实现
import torch.nn.functional as F	# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim	# 实现各种优化算法的包
from torchvision import datasets, transforms
from collections import OrderedDict
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

###### parameter ######
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stride = 1
padding=1
batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
log_interval = 1
torch.manual_seed(1)
np.random.seed(1)
print("device {}".format(device))

###### net model ######
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3,8,3,stride,padding)),
          ('relu1', nn.ReLU()),
          ('pool1', nn.MaxPool2d(2)),
          ('conv2', nn.Conv2d(8,12,3,stride,padding)),
          ('relu2', nn.ReLU()),
          ('pool2', nn.MaxPool2d(2)),
          ('conv3', nn.Conv2d(12,16,3,stride,padding)),
          ('relu3', nn.ReLU()),
          ('unsample1', nn.Upsample(scale_factor=(2,2))),
          ('convtranspose1', nn.ConvTranspose2d(16,12,3,stride,padding)),
          ('relu4', nn.ReLU()),
          ('unsample2', nn.Upsample(scale_factor=(2,2))),
          ('convtranspose2', nn.ConvTranspose2d(12,3,3,stride,padding)),
          ('relu5', nn.ReLU()),
        ])).to(device)

###### transform ######
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
transform = transforms.Compose([transforms.ToTensor(),normalize])
augment_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize])

###### dataset ######
train_dataset = datasets.CIFAR10(root='/content/drive/My Drive/Colab Notebooks/data/',
                                 train=True,download=True, transform=augment_transform)
test_dataset = datasets.CIFAR10(root='/content/drive/My Drive/Colab Notebooks/data/',
                                train=False,download=True, transform=transform)
valid_dataset = datasets.CIFAR10(root='/content/drive/My Drive/Colab Notebooks/data/',
                                 train=True,download=True, transform=transform)

###### dataset split ######
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.125 * num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

###### dataset loader ######
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          sampler=train_sampler,num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                           sampler=valid_sampler,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



###### optimizaion ######
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

###### train function ######
def train(model, device, train_loader, optimizer, epoch):
    model.train() # set the model to train mode
    for batch_idx, (data, target) in enumerate(train_loader): # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device) # 将数据存储CPU或者GPU
        optimizer.zero_grad() # 清除所有优化的梯度
        output = model(data)  # 喂入数据并前向传播获取输出
        # print("data.shape {}".format(data.shape))
        # print("output.shape {}".format(output.shape))
        loss = criterion(output, data) # 调用损失函数计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        if batch_idx % log_interval == 0: # 根据设置的显式间隔输出训练日志
          print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx , len(train_loader),
              100. * batch_idx / len(train_loader), loss.item()),end = '')

###### test function ######
def test(model, device, test_loader):
    model.eval() # set the model to test mode
    test_loss = 0
    with torch.no_grad(): # 禁用梯度计算
        for data, target in test_loader: # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, data) # sum up batch loss
    test_loss /= len(test_loader.dataset)/test_batch_size
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

###### validate function ######
def validate(model, device, valid_loader):
    model.eval() # set the model to test mode
    valid_loss = 0
    with torch.no_grad(): # 禁用梯度计算
        for data, target in valid_loader: # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device)
        output = model(data)
        valid_loss += criterion(output, data) # sum up batch loss
    valid_loss /= len(valid_loader.dataset)/batch_size
    print('\nValid set: Average loss: {:.4f}'.format(valid_loss))

###### main function ######
def main():
    for epoch in range(1, epochs+1): # 循环调用train() and test() 进行epochs迭代
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        # validate(model, device, valid_loader)
        torch.save(model.state_dict(), '/content/drive/My Drive/Colab Notebooks/data/params.pkl')

###### run the main function ######
main()
