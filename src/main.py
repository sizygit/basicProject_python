# This is a sample Python script.
import network
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
data = np.load('../data/MNIST_DATA/MNIST_DATA.npy', allow_pickle=True)
data = data.item()
train_data, test_data = data['train_data'], data['test_data']
print('size is', len(train_data), len(test_data))

std_net = network.Network([784, 30, 10])
std_net.SGD(train_data, 30, 10, 3.0, test_data=test_data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
