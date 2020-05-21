# -*- coding: UTF-8 -*-
from imutils import paths
import numpy as np
import  cv2 as cv
from torch.utils.data import Dataset
import time
import os
from model.cnn_model import LeNet5,CHARS,CHARS_DICT
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ListDataset(Dataset):
    '''
    用于加载训练数据
    '''
    def __init__(self, list_path,size):
        self.dir=list_path
        self.img_paths=[]
        self.img_paths += [el for el in paths.list_images(list_path)]
        self.size=size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        # print(basename,imgname)
        label = imgname.split("_")[1]
        label = CHARS_DICT[label]
        file = self.dir + basename
        image = cv.imread(file)
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image = cv.resize(image, self.size)
        image=image.reshape(1, 32, 32)
        image = np.asarray(image, 'float32')
        image = torch.from_numpy(image)
        input_img= image
        return input_img,label

def validate(iter):
    '''
    func:网络的验证集计算
    '''
    total_correct = 0
    avg_loss = 0.0
    global best_acc
    for i, (images, labels) in enumerate(data_val_loader):
        images = images
        labels = labels
        #print(images.shape)
        #print(labels.shape)
        with torch.no_grad():
            '''
            预测验证集的网络计算结果，不用计算梯度，可以用于减少内存的
            '''
            output = net(images) #验证集的网络输出结果
        # 同样计算训练集网络训练的结果和真实结果的loss，
        # 这里采用的cross_entropy这个loss计算
        avg_loss += F.cross_entropy(output, labels).sum()

        pred = output.detach().max(1)[1]  # detach cell from the model graph

        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_val_loader)
    acc_test= float(total_correct) / len(data_val)
    writer.add_scalar("Loss/test",avg_loss,iter)
    writer.add_scalar("Accuracy/test",acc_test,iter)
    print('epoch:%d ,Validation Avg. Loss: %f, Accuracy: %f' % (iter+1,
    avg_loss.detach().cpu().item(), float(total_correct) / len(data_val)))
    if float(total_correct) / len(data_val) > best_acc:
        '''
        保存最好准确率的模型训练参数
        '''
        best_acc = float(total_correct) / len(data_val)
        torch.save(net.state_dict(), './model/best_model.pt')

def train(epoch):
    avg_loss = 0
    for i in range(epoch):
        total_train_cornum = 0
        for images, labels in data_train_loader:
            '''
            把训练数据一个个放入网络
            '''
            #print(images.size())
            optimizer.zero_grad() #将所有参数和反向传播器的梯度缓冲区归零
            output = net(images) #把图片放进去网络进行计算
            #print(output.shape)
            loss = F.cross_entropy(output, labels) #计算每一个样本在网络中的计算值和样本ground truth的loss，这里采用的cross_entropy这个loss计算

            avg_loss += loss.detach().cpu().item() #用于计算每一个epoch的训练集的累积loss，便于后面计算平均loss
            loss.backward() #反向传播计算梯度
            optimizer.step() #网络的weight更新
            pred_train = output.argmax(dim=1).eq(labels).sum().item()#网络训练的正确对应个数
            #print(pred_train.eq(labels.view_as(pred_train)).size())
            total_train_cornum += pred_train

        # 计算1个Epoch的平均 Training Loss
        avg_loss = avg_loss / len(data_train_loader)
        accury_train = float(total_train_cornum)/ len(data_train)
        writer.add_scalar("Loss/train",avg_loss,i)
        writer.add_scalar("Accuracy/train", accury_train, i)
        print("epoch:{} Mean Trainning Loss:{:.4f},Train Accuracy :{}".format(i+1,avg_loss,accury_train))
        validate(iter=i)

if __name__ =="__main__":
    torch.manual_seed(2)
    if not os.path.exists('./model/'):
        os.makedirs('./model/')
    # prepare data
    best_acc = 0
    tarin_bsize = 200  # 训练集的batch size
    val_bsize = 1024  # 验证集的batch size
    data_train = ListDataset("./train_data/", size=(32, 32))
    data_val = ListDataset("./test_data/", size=(32, 32))
    data_train_loader = DataLoader(data_train, batch_size=tarin_bsize, shuffle=True, num_workers=8)
    data_val_loader = DataLoader(data_val, batch_size=val_bsize, num_workers=8)
    start_time=time.time()
    best_acc = 0            #最佳准确率
    learning_rate = 0.001   #学习率
    epoch_num = 50          #回合数
    net = LeNet5()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0) #optim方法用于更新网络中的weight，参数lr是学习率
    writer=SummaryWriter(log_dir="runs/"+"lr="+str(learning_rate)+" _epoch_num"+str(epoch_num)+"_tr-bz="+str(tarin_bsize))
    train(epoch_num)
    print("best accuracy:",best_acc)
    print(f"Run time{round(time.time()-start_time,2)}")





