import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from PIL import Image
from torch import optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR


class Datasetfloder(Dataset):
    """sparse correspondences dataset"""
    def __init__(self, istrain="train"):
        data = np.load('../data/MNIST_DATA.npy', allow_pickle=True)
        data = data.item()
        self.train = istrain
        train_data, test_data = data['train_data'], data['test_data']
        if self.train == 'train':
            train_data = [(s[0], np.argmax(s[1][:, 0])) for s in train_data]
            self.data = train_data
        else:
            self.data = test_data
        self.lens = len(self.data)

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = x.reshape((1, 28, 28))
        return x, y


class Networkv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # in_channels: int,out_channels: int,
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # kernel_size: _size_2_t, stride: _size_2_t = 1,
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10816, 128)    # need to change
        self.fc2 = nn.Linear(128, 2)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True)  # pool layer

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizerm, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizerm.zero_grad()  # set optimizer 's grad is zero
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()  # 后向传递
        optimizerm.step()  # 更新优化器
        if batch_idx % log_interval == 0:
            print('train epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdims=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\n Test set: Average loss：{:.4f},Accurancy:{}/{}({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)  # link the path
        self.img_path_list = os.listdir(self.path)  # generate a list containing the names of the files in the directory

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((30, 30)),
                                                    torchvision.transforms.ToTensor()])
        img = transform(img)
        label = int(self.label_dir)  # convert the str to int
        return img, label

    def __len__(self):
        return len(self.img_path_list)



def main():
    torch.manual_seed(1)
    log_interval = 10
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")
    train0 = MyData("../data/fgdData/train", "0")
    train1 = MyData("../data/fgdData/train", "1")
    train_dataset = train0 + train1
    test0 = MyData("../data/fgdData/test", "0")
    test1 = MyData("../data/fgdData/test", "1")
    test_dataset = test0 + test1
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True, batch_size=32, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=False, batch_size=10, drop_last=False)
    model = Networkv2().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.8)  # 优化器
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)  #
    for epoch in range(1, 14 + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)
        scheduler.step()

    if True:
        torch.save(model.state_dict(), "../faguangdian.pt")



main()







