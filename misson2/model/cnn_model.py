import torch
import torch.nn as nn
import  numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import cv2 as cv
CHARS = [
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LeNet5(nn.Module):
    """
    Input - 1x32x32，网络的输入大小是32x32，严格遵循这个标准，假如不是这个尺寸要resize成这个尺寸
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    F7 - 10 (Output)

    Feel free to try different filter numbers
    Output:输出的是一个tensor的张量，size看最后一层网络的output
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 16, kernel_size=(5,5),stride=1)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('c3', nn.Conv2d(16, 50, kernel_size=(5,5),stride=1)),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('c5', nn.Conv2d(50, 120, kernel_size=(5,5), stride=1)),
            ('relu5', nn.ReLU())

        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 34))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        #print("1",output.size())
        output = output.view(img.size(0), -1)
        #print("2",output.size(),img.size())
        output = self.fc(output)
        #print("3",output.size())

        return output

def create_cnn(img,model_dir="./model/best_model.pt"):
    net= LeNet5()
    net.load_state_dict(torch.load(model_dir))
    image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (32,32))
    image=image.reshape(1,1, 32, 32)
    image = np.asarray(image, 'float32')
    image = torch.from_numpy(image)
    with torch.no_grad():
        output=net(image)
        pred = output.detach().max(1)[1]
        #print(CHARS[int(pred.item())])
        return CHARS[int(pred.item())]