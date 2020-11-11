# %%

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
batch_size = 64
input_gamma = [0.1, 0.5, 1.5, 3]
# input_gamma = [1.0]
data_path = './'
# data_path = '/content/drive/My Drive/Colab Notebooks/'
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


# %%

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
    # 0-15 reserved footroom
    # 236-255 reserved headroom
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


# %%

###### net model for reconstructure ######
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(len(input_gamma), 16, 3, stride=stride, padding=padding)),
    ('relu1', nn.LeakyReLU(l_relu_slope)),

    ('conv2', nn.Conv2d(16, 16, 3, stride=stride, padding=padding)),
    ('relu2', nn.LeakyReLU(l_relu_slope)),

    ('conv3', nn.Conv2d(16, 16, 3, stride=2, padding=padding)),
    ('relu3', nn.LeakyReLU(l_relu_slope)),

    ('conv4', nn.Conv2d(16, 32, 3, stride=stride, padding=padding)),
    ('relu4', nn.LeakyReLU(l_relu_slope)),

    ('conv5', nn.Conv2d(32, 32, 3, stride=2, padding=padding)),
    ('relu5', nn.LeakyReLU(l_relu_slope)),

    ('unsample1', nn.Upsample(scale_factor=(2, 2))),

    ('convtranspose1', nn.ConvTranspose2d(32, 32, 3, stride, padding)),
    ('relu6', nn.LeakyReLU(l_relu_slope)),

    ('convtranspose2', nn.ConvTranspose2d(32, 16, 3, stride, padding)),
    ('relu7', nn.LeakyReLU(l_relu_slope)),

    ('unsample2', nn.Upsample(scale_factor=(2, 2))),

    ('convtranspose3', nn.ConvTranspose2d(16, 8, 3, stride, padding)),
    ('relu8', nn.LeakyReLU(l_relu_slope)),

    ('convtranspose4', nn.ConvTranspose2d(8, 2, 3, stride, padding)),
    ('relu9', nn.Sigmoid()),
])).to(device)

# %%

###### transform ######
# normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                  std=[0.2023, 0.1994, 0.2010])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                  std=[0.5, 0.5, 0.5])
transform = transforms.Compose([transforms.ToTensor()])
augment_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# %%

###### dataset ######
train_dataset = datasets.CIFAR10(root=data_path + 'data/',
                                 train=True, download=True, transform=augment_transform)
test_dataset = datasets.CIFAR10(root=data_path + 'data/',
                                train=False, download=True, transform=transform)
valid_dataset = datasets.CIFAR10(root=data_path + 'data/',
                                 train=True, download=True, transform=transform)

# %%

###### dataset split ######
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.125 * num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# %%

###### dataset loader ######
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          shuffle=False, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%

###### optimizaion ######
criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)


# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.RMSprop(model.parameters(), lr=lr)


# %%

###### train function ######
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # set the model to train mode
    for batch_idx, (data, target) in enumerate(train_loader):  # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device)  # 将数据存储CPU或者GPU
        input_size = data.size()

        optimizer.zero_grad()  # 清除所有优化的梯度
        gray_data = rgb_to_grayscale(data)
        input = Variable(data.data.new(input_size[0], len(input_gamma), input_size[2], input_size[3]))

        for i in range(len(input_gamma)):
            gamma = input_gamma[i]
            input[:, i, :, :] = torchvision.transforms.functional.adjust_gamma(gray_data, gamma)[:, 0, :, :]

        output = model(input)  # 喂入数据并前向传播获取输出

        # r_output_image = Variable(data.data.new(*input_size))
        #
        # r_output_image[:, 0, :, :] = gray_data[:, 0, :, :]
        # r_output_image[:, 1, :, :] = output[:, 0, :, :]
        # r_output_image[:, 2, :, :] = output[:, 1, :, :]

        # r_output_image = ycbr_to_rgb(r_output_image)

        r_output_image = rgb_to_cbcr(data)

        # loss = criterion(r_output_image, output)  # 调用损失函数计算损失
        loss = criterion(output, r_output_image)  # 调用损失函数计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        if batch_idx % log_interval == 0:  # 根据设置的显式间隔输出训练日志
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item() / batch_size), end='')
    loss /= len(train_loader)
    return loss.item()


# %%

###### test function ######
def test(model, device, test_loader):
    model.eval()  # set the model to test mode
    test_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:  # 从数据加载器迭代一个batch的数据
            data, target = data.to(device), target.to(device)
            input_size = data.size()

            optimizer.zero_grad()  # 清除所有优化的梯度
            gray_data = rgb_to_grayscale(data)
            input = Variable(data.data.new(input_size[0], len(input_gamma), input_size[2], input_size[3]))

            for i in range(len(input_gamma)):
                gamma = input_gamma[i]
                input[:, i, :, :] = torchvision.transforms.functional.adjust_gamma(gray_data, gamma)[:, 0, :, :]

            output = model(input)  # 喂入数据并前向传播获取输出
            # r_output_image = Variable(data.data.new(*input_size))

            # r_output_image = Variable(data.data.new(*input_size))
            #
            # r_output_image[:, 0, :, :] = gray_data[:, 0, :, :]
            # r_output_image[:, 1, :, :] = output[:, 0, :, :]
            # r_output_image[:, 2, :, :] = output[:, 1, :, :]

            # r_output_image = ycbr_to_rgb(r_output_image)

            r_output_image = rgb_to_cbcr(data)

            test_loss += criterion(output, r_output_image)  # sum up batch loss
        test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_loss.item()


# %%

###### validate function ######
def validate(model, device, valid_loader):
    model.train(False)
    valid_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in valid_loader:  # 从数据加载器迭代一个batch的数据
            data, target = data.to(device), target.to(device)
            input_size = data.size()

            optimizer.zero_grad()  # 清除所有优化的梯度
            gray_data = rgb_to_grayscale(data)
            input = Variable(data.data.new(input_size[0], len(input_gamma), input_size[2], input_size[3]))

            for i in range(len(input_gamma)):
                gamma = input_gamma[i]
                input[:, i, :, :] = torchvision.transforms.functional.adjust_gamma(gray_data, gamma)[:, 0, :, :]

            output = model(input)  # 喂入数据并前向传播获取输出
            # r_output_image = Variable(data.data.new(*input_size))

            # r_output_image = Variable(data.data.new(*input_size))
            #
            # r_output_image[:, 0, :, :] = gray_data[:, 0, :, :]
            # r_output_image[:, 1, :, :] = output[:, 0, :, :]
            # r_output_image[:, 2, :, :] = output[:, 1, :, :]

            # r_output_image = ycbr_to_rgb(r_output_image)

            r_output_image = rgb_to_cbcr(data)
            valid_loss += criterion(output, r_output_image)  # sum up batch loss
        valid_loss /= len(valid_loader)
    print('\nValid set: Average loss: {:.4f}'.format(valid_loss))
    return valid_loss.item()


# %%

losses = {'train': [], 'val': [], 'test': []}


###### main function ######
def main():
    for epoch in range(1, epochs + 1):  # 循环调用train() and test() 进行epochs迭代
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, test_loader)
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        torch.save(model.state_dict(), data_path + 'data/params.pkl')
    test_loss = test(model, device, test_loader)
    losses['test'].append(test_loss)


# %% run main

main()


# %%

def imshow(img, cmap=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
    # plt.savefig(data_path+'test_image.png')
    plt.show()


def imshow_yuv(img, n_img=3):
    for i in range(n_img):
        p = img[i, :, :].reshape((1, 32, 32))
        plt.imshow(np.transpose(p, (1, 2, 0)), cmap='gray')
        plt.show()


# %%
dataiter = iter(test_loader)
# dataiter = iter(train_loader)

# %%
# get some random training images
images, labels = dataiter.next()
input = None
out = None
data = None
with torch.no_grad():  # 禁用梯度计算
    data, target = images.to(device), labels.to(device)
    input_size = data.size()

    gray_data = rgb_to_grayscale(data)
    input = Variable(data.data.new(input_size[0], len(input_gamma), input_size[2], input_size[3]))

    for i in range(len(input_gamma)):
        gamma = input_gamma[i]
        input[:, i, :, :] = torchvision.transforms.functional.adjust_gamma(gray_data, gamma)[:, 0, :, :]
    output = model(input)  # 喂入数据并前向传播获取输出
    out = rgb_to_cbcr(data)

# %% Print out an example out image

# test_image = data[0].cpu().numpy().transpose((1, 2, 0))
test_input = rgb_to_ycbcr(data)
test_output = ycbr_to_rgb(test_input)

input_size = data[:1].size()
r_input_image = Variable(data[:1].data.new(*input_size))

r_input_image[:1, 0, :, :] = gray_data[:1, 0, :, :]
r_input_image[:1, 1, :, :] = output[:1, 0, :, :]
r_input_image[:1, 2, :, :] = output[:1, 1, :, :]

r_input_image = ycbr_to_rgb(r_input_image)

# imshow_yuv(test_input[0].cpu(), n_img=3)
# plt.imshow(test_output[0].transpose(1, 2).T.cpu())
# plt.imshow(test_output[0].transpose(1, 2).T.cpu())

plt.imshow(gray_data[0].transpose(1, 2).T.cpu(), cmap='gray')
plt.show()

plt.imshow(r_input_image[0].transpose(1, 2).T.cpu())
plt.show()

plt.imshow(data[0].transpose(1, 2).T.cpu())
plt.show()


# %%
# Plot losses
losses['train'] = np.array(losses['train'])
losses['val'] = np.array(losses['val'])
x_epochs = np.arange(1, epochs + 1)

# plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])

plt.plot(x_epochs, losses['train'])
plt.plot(x_epochs, losses['val'])

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.xticks(np.arange(epochs + 1))
plt.yticks(np.arange(0.0, 1.0, 0.1))

plt.legend(['Train loss', 'Validation loss'], loc='upper left')
plt.grid(True)
plt.title("Training loss of the basic autoencoder")

plt.show()