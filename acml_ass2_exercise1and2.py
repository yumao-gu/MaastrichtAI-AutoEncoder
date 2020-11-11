import torch
import torch.nn as nn  # 各种层类型的实现
import torch.nn.functional as F  # 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim  # 实现各种优化算法的包
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from collections import OrderedDict
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% parameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stride = 1
padding = 1
batch_size = 16
input_gamma = [1, 0.5, 1.5, 3]
# data_path = './'
data_path = '/content/drive/My Drive/Colab Notebooks/'
test_batch_size = 1000
epochs = 10
lr = 0.001
momentum = 0.5
rgb2ycbrform = torch.tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
ycbr2rgbform = torch.tensor([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
l_relu_slope = 0.01
log_interval = 1
torch.manual_seed(1)
np.random.seed(1)
print("device {}".format(device))

def rgb_to_Y(input_img):
    return 16 + ((input_img[:, 0, :, :] * 65.481)
                 + (input_img[:, 1, :, :] * 128.553)
                 + (input_img[:, 2, :, :] * 24.966))

def rgb_to_Cb(input_img):
    return 128 + ((input_img[:, 0, :, :] * (-37.797))
                  - (input_img[:, 1, :, :] * 74.203)
                  + (input_img[:, 2, :, :] * 112.0))

def rgb_to_Cr(input_img):
    return 128 + ((input_img[:, 0, :, :] * 112.0)
                  - (input_img[:, 1, :, :] * 93.786)
                  - (input_img[:, 2, :, :] * 18.214))

def ycbcr_to_R(input_img):
    return (255 / 219) * (input_img[:, 0, :, :] - 16) \
           + (255 / 224) * 1.402 * (input_img[:, 2, :, :] - 128)

def ycbcr_to_G(input_img):
    return (255 / 219) * (input_img[:, 0, :, :] - 16) \
           - (255 / 224) * 1.772 * (0.114 / 0.587) * (input_img[:, 1, :, :] - 128) \
           - (255 / 224) * 1.402 * (0.299 / 0.587) * (input_img[:, 2, :, :] - 128)

def ycbr_to_B(input_img):
    return (255 / 219) * (input_img[:, 0, :, :] - 16) \
           + (255 / 224) * 1.772 * (input_img[:, 1, :, :] - 128)

def rgb_to_grayscale(input_img):
    input_size = input_img.size()
    output = Variable(input_img.data.new(input_size[0], 1, input_size[2], input_size[3]))
    output[:, 0, :, :] = ((rgb_to_Y(input_img) - 16) / 219)
    return output

def rgb_to_ycbcr(input_img):
    output = Variable(input_img.data.new(*input_img.size()))
    r_input_image = input_img
    cbcr = rgb_to_cbcr(input_img)
    # formulas from https://en.wikipedia.org/wiki/YCbCr for analog RGB to YCBR.
    output[:, 0, :, :] = ((rgb_to_Y(r_input_image) - 16) / 219)
    output[:, 1, :, :] = cbcr[:, 0, :, :]
    output[:, 2, :, :] = cbcr[:, 1, :, :]
    return output

def rgb_to_cbcr(input_img):
    input_size = input_img.size()
    output = Variable(input_img.data.new(input_size[0], 2, input_size[2], input_size[3]))
    r_input_image = input_img
    # formulas from https://en.wikipedia.org/wiki/YCbCr for analog RGB to YCBR.
    # 0-15 reserved footroom,236-255 reserved headroom
    output[:, 0, :, :] = ((rgb_to_Cb(r_input_image) - 16) / 224)
    output[:, 1, :, :] = ((rgb_to_Cr(r_input_image) - 16) / 224)
    return output

def ycbr_to_rgb(input_img):
    input_size = input_img.size()
    r_input_image = Variable(input_img.data.new(*input_size))
    output = Variable(input_img.data.new(*input_size))
    r_input_image[:, 0, :, :] = (input_img[:, 0, :, :] * 219) + 16
    r_input_image[:, 1, :, :] = (input_img[:, 1, :, :] * 224) + 16
    r_input_image[:, 2, :, :] = (input_img[:, 2, :, :] * 224) + 16
    # formulas from https://en.wikipedia.org/wiki/YCbCr for YCbCr
    output[:, 0, :, :] = ycbcr_to_R(r_input_image)
    output[:, 1, :, :] = ycbcr_to_G(r_input_image)
    output[:, 2, :, :] = ycbr_to_B(r_input_image)
    return output / 255

###### net model for reconstructure ######
# model = nn.Sequential(OrderedDict([
#     ('conv1', nn.Conv2d(4, 16, 3, stride=stride, padding=padding)),
#     ('relu1', nn.LeakyReLU(l_relu_slope)),
#     ('conv2', nn.Conv2d(16, 16, 3, stride=stride, padding=padding)),
#     ('relu2', nn.LeakyReLU(l_relu_slope)),
#     ('conv3', nn.Conv2d(16, 16, 3, stride=2, padding=padding)),
#     ('relu3', nn.LeakyReLU(l_relu_slope)),
#     ('conv4', nn.Conv2d(16, 32, 3, stride=stride, padding=padding)),
#     ('relu4', nn.LeakyReLU(l_relu_slope)),
#     ('conv5', nn.Conv2d(32, 32, 3, stride=2, padding=padding)),
#     ('relu5', nn.LeakyReLU(l_relu_slope)),
#     ('unsample1', nn.Upsample(scale_factor=(2, 2))),
#     ('convtranspose1', nn.ConvTranspose2d(32, 32, 3, stride, padding)),
#     ('relu6', nn.LeakyReLU(l_relu_slope)),
#     ('convtranspose2', nn.ConvTranspose2d(32, 16, 3, stride, padding)),
#     ('relu7', nn.LeakyReLU(l_relu_slope)),
#     ('unsample2', nn.Upsample(scale_factor=(2, 2))),
#     ('convtranspose3', nn.ConvTranspose2d(16, 8, 3, stride, padding)),
#     ('relu8', nn.LeakyReLU(l_relu_slope)),
#     ('convtranspose4', nn.ConvTranspose2d(8, 2, 3, stride, padding)),
#     ('relu9', nn.Sigmoid()),
# ])).to(device)

###### net model for reconstructure ######
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3,12,3,stride,padding)),
          ('relu1', nn.LeakyReLU(l_relu_slope)),
          ('pool1', nn.MaxPool2d(2)),
          # ('pool1', nn.Conv2d(8,8,3,2,padding)),
          # ('conv2', nn.Conv2d(8,12,3,stride,padding)),
          # ('relu2', nn.LeakyReLU(l_relu_slope)),
          # ('pool2', nn.MaxPool2d(2)),
          # ('pool2', nn.Conv2d(12,12,3,2,padding)),
          ('conv3', nn.Conv2d(12,16,3,stride,padding)),
          ('relu3', nn.LeakyReLU(l_relu_slope)),
          # ('pool0', nn.MaxPool2d(2)),
          # ('conv0', nn.Conv2d(16,32,3,stride,padding)),
          # ('relu0', nn.LeakyReLU(l_relu_slope)),
          # ('unsample0', nn.Upsample(scale_factor=(2,2))),
          # ('convtranspose0', nn.ConvTranspose2d(32,16,3,stride,padding)),
          # ('unsample1', nn.Upsample(scale_factor=(2,2))),
          # ('convtranspose1', nn.ConvTranspose2d(16,12,3,stride,padding)),
          # ('relu4', nn.LeakyReLU(l_relu_slope)),
          ('unsample2', nn.Upsample(scale_factor=(2,2))),
          ('convtranspose2', nn.ConvTranspose2d(16,3,3,stride,padding)),
          ('sigmoid', nn.Sigmoid()),
        ])).to(device)

###### transform ######
# normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                  std=[0.2023, 0.1994, 0.2010])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                  std=[0.5, 0.5, 0.5])
transform = transforms.Compose([transforms.ToTensor()])
augment_transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

###### dataset ######
train_dataset = datasets.CIFAR10(root=data_path + 'data/',
                                 train=True, download=True, transform=augment_transform)
test_dataset = datasets.CIFAR10(root=data_path + 'data/',
                                train=False, download=True, transform=transform)
valid_dataset = datasets.CIFAR10(root=data_path + 'data/',
                                 train=True, download=True, transform=transform)

###### dataset split ######
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.05 * num_train))
# np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

###### dataset loader ######
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          shuffle=False, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###### optimizaion ######
###### https://zhuanlan.zhihu.com/p/29920135
###### https://blog.csdn.net/bvl10101111/article/details/72616378
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# optimizer = optim.RMSprop(model.parameters(), lr=lr)

###### train function ######
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # set the model to train mode
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):  # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device)  # 将数据存储CPU或者GPU
        output = model(data)  # 喂入数据并前向传播获取输出
        optimizer.zero_grad()  # 清除所有优化的梯度
        loss = criterion(data, output)  # 调用损失函数计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        if batch_idx % log_interval == 0:  # 根据设置的显式间隔输出训练日志
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()), end='')
        train_loss += loss
    train_loss /= len(train_loader)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss.item()

###### test function ######
def test(model, device, test_loader):
    model.eval()  # set the model to test mode
    test_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:  # 从数据加载器迭代一个batch的数据
            data, target = data.to(device), target.to(device)
            output = model(data)  # 喂入数据并前向传播获取输出
            
            optimizer.zero_grad()  # 清除所有优化的梯度
            test_loss += criterion(data, output)  # sum up batch loss
        test_loss /= len(test_loader)
        print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss.item()

###### validate function ######
def validate(model, device, valid_loader):
    model.train(False)
    valid_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in valid_loader:  # 从数据加载器迭代一个batch的数据
            data, target = data.to(device), target.to(device)
            output = model(data)  # 喂入数据并前向传播获取输出

            optimizer.zero_grad()  # 清除所有优化的梯度
            valid_loss += criterion(data, output)  # sum up batch loss
        valid_loss /= len(valid_loader)
        print('Valid set: Average loss: {:.4f}'.format(valid_loss))
    return valid_loss.item()

losses = {'train': [], 'val': [], 'test': []}

###### main function ######
def main():
    for epoch in range(1, epochs + 1):  # 循环调用train() and test() 进行epochs迭代
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, valid_loader)
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        torch.save(model.state_dict(), data_path + 'data/params.pkl')
        test_loss = test(model, device, test_loader)
        losses['test'].append(test_loss)
    print("loss['train'] {}".format(losses['train']))
    print("loss['val'] {}".format(losses['val']))
    print("loss['test'] {}".format(losses['test']))

    x = np.arange(0, len(losses['test']), 1)
    y=[]
    for data in losses['train']:
        y.append(data)
    y = np.array(y)
    plt.plot(x, y, color='r', linestyle="-", linewidth=1, label='train')

    y=[]
    for data in losses['val']:
        y.append(data)
    y = np.array(y)
    plt.plot(x, y, color='y', linestyle="-", linewidth=1, label='val')

    y=[]
    for data in losses['test']:
        y.append(data)
    y = np.array(y)
    plt.plot(x, y, color='g', linestyle="-", linewidth=1, label='test')
    plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.95))
    plt.title("loss J")
    plt.show()

# %% run main
main()

###### visualization ######
def imshow(img, cmap=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
    # plt.savefig(data_path+'test_image.png')
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
input = None
out = None
data = None
gray_data = None
model.eval()
with torch.no_grad():  # 禁用梯度计算
    data, target = images.to(device), labels.to(device)
    output = model(data)  # 喂入数据并前向传播获取输出

plt.imshow(images[0].transpose(1, 2).T.cpu())
plt.show()
# imshow(torchvision.utils.make_grid(gray_data[0].cpu()))
plt.imshow(output[0].transpose(1, 2).T.cpu())
plt.show()
