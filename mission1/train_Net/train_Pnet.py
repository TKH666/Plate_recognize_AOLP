import sys
import os
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset
from data.Data_Loading import ListDataset
from model.model_net import PNet
import time
import copy
import torch.nn as nn

def weights_init(m):
    '''
    初始化网络权重
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

train_path = './data/trainPnet/anno_store/imglist_anno_pnet.txt'
val_path = './data/valPnet/anno_store/imglist_anno_pnet_val.txt'
batch_size = 64
dataloaders = {'train': torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True),
               'val': torch.utils.data.DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True)}
dataset_sizes = {'train': len(ListDataset(train_path)), 'val': len(ListDataset(val_path))}
print('训练集的大小为 : {}'.format(len(ListDataset(train_path))))
print('验证集的大小为 : {}'.format(len(ListDataset(val_path))))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载模型并且初始化网络中的权重参数
model = PNet(is_train=True).to(device)
model.apply(weights_init)
print("Pnet 加载完成")

#模型最小loss储存网络权重
best_model_weights = copy.deepcopy(model.state_dict())
best_accuracy = 0.0
best_loss = 100

#分别定义不同任务的loss
#车牌二分类问题loss采用交叉熵
loss_cls = nn.CrossEntropyLoss()

#车牌的bounding box回归问题采用欧式距离的方差
loss_offset = nn.MSELoss()

#训练过程产生的log文件
train_logging_file = './train_Net/Pnet_train_logging.txt'

#优化器设定
optimizer = torch.optim.Adam(model.parameters())
since = time.time()




num_epochs = 16
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-' * 10)

    # 每一个epoch要训练和验证模型
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 模型转换为训练模式
        else:
            model.eval()  #  模型转换为验证模式

        running_loss, running_loss_cls, running_loss_offset = 0.0, 0.0, 0.0
        running_correct = 0.0
        running_gt = 0.0

        # 训练开始，把batch数据全迭代一次
        for i_batch, sample_batched in enumerate(dataloaders[phase]):

            input_images, gt_label, gt_offset = sample_batched['input_img'], sample_batched[
                'label'], sample_batched['bbox_target']
            input_images = input_images.to(device)
            gt_label = gt_label.to(device)
            # print('gt_label is ', gt_label)
            gt_offset = gt_offset.type(torch.FloatTensor).to(device)
            # print('gt_offset shape is ',gt_offset.shape)

            # 梯度参数0初始化
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                pred_offsets, pred_label = model(input_images)
                pred_offsets = torch.squeeze(pred_offsets)
                pred_label = torch.squeeze(pred_label)
                # 车牌二分类问题的lable计算，车牌的二分类问题中，只需要用到label=0(iou<0.3) 和 1 (iou>0.6)的样本
                # 只有0和1的label才是二分类的预测结果，筛选label >=0的训练样本
                mark_cls = torch.ge(gt_label, 0) #label>=0
                valid_gt_label = gt_label[mark_cls] #subset取出训练样本中 label= 0，1的样本
                valid_pred_label = pred_label[mark_cls] #pnet对label=0/1 的训练样本产生对应概率

                # bounding box回归问题的计算，坐标的回归问题用到label = 1(iou>0.6)和-1(0.4<=iou<=0.6)的样本

                mark_0 = torch.eq(gt_label, 0) # 筛选label=0的样本
                mark_no0 = torch.eq(mark_0, 0) #subset反选取出label=1和-1的样本
                valid_gt_offset = gt_offset[mark_no0] #反选出label =1/-1的训练样本的offset
                valid_pred_offset = pred_offsets[mark_no0] #pnet对label=-1/1 的训练样本产生对应offset

                loss = torch.tensor(0.0).to(device)
                cls_loss, offset_loss = 0.0, 0.0
                eval_correct = 0.0
                num_gt = len(valid_gt_label)

                if len(valid_gt_label) != 0:
                    #车牌分类的loss计算，分类问题loss权重是0.02
                    loss += 0.02*loss_cls(valid_pred_label, valid_gt_label)
                    cls_loss = loss_cls(valid_pred_label, valid_gt_label).item()
                    pred = torch.max(valid_pred_label, 1)[1]
                    eval_correct = (pred == valid_gt_label).sum().item()

                if len(valid_gt_offset) != 0:
                    #bounding box回归loss计算，回归问题的loss权重是0.6
                    loss += 0.6*loss_offset(valid_pred_offset, valid_gt_offset)
                    offset_loss = loss_offset(valid_pred_offset, valid_gt_offset).item()


                if phase == 'train':
                    #loss的反向梯度传播计算，并且最小化loss
                    loss.backward()
                    optimizer.step()

                # 统计相关信息
                running_loss += loss.item()*batch_size
                running_loss_cls += cls_loss*batch_size
                running_loss_offset += offset_loss*batch_size
                running_correct += eval_correct
                running_gt += num_gt

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_loss_cls = running_loss_cls / dataset_sizes[phase]
        epoch_loss_offset = running_loss_offset / dataset_sizes[phase]
        epoch_accuracy = running_correct / (running_gt + 1e-16)

        print('{} Loss: {:.4f} 车牌分类accuracy: {:.4f} 车牌分类Loss: {:.4f} bounding box回归 Loss: {:.4f}'
              .format(phase, epoch_loss, epoch_accuracy, epoch_loss_cls, epoch_loss_offset))
        with open(train_logging_file, 'a') as f:
            f.write('{} Loss: {:.4f} 车牌分类accuracy: {:.4f} 车牌分类Loss: {:.4f} bounding box回归 Loss: {:.4f}'
                    .format(phase, epoch_loss, epoch_accuracy, epoch_loss_cls, epoch_loss_offset)+'\n')
        f.close()

        # 复制模型保存
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict()) #保存最小loss的模型的权重

time_elapsed = time.time() - since
print('训练完成，耗费 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best loss: {:4f}'.format(best_loss))

model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), './model/pnet_Weights')