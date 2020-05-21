import numpy as np

def img_preprocess(img):
    """图像的预处理，增加一维，改变shape，而且对图像进行均值和归一化.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img[:,:,::-1]
    img = np.asarray(img, 'float32')
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5)*0.0078125 #去均值，归一化

    return img

def nms(boxes, overlap_threshold=0.5, mode='union'):
    """非极大值抑制算法

    Arguments:
        boxes: float numpy array of shape [n, 5],
            每一行是 (xmin, ymin, xmax, ymax, score).
        overlap_threshold: float number.用于调整重叠iou的阈值
        mode: 'union' 或者 'min' 计算iou的模型

    Returns:
        返回一个list，选取select的box的索引
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # 用于储存候选框的下标的list
    pick = []

    # grab the coordinates of the bounding boxes
    #每一个像素的候选框的坐标以及是车牌的概率
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    #print("s1",x1.shape)
    #area是每一个候选框的面积
    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    #根据概率大小进行排序，有小到大排序，返回下标
    ids = np.argsort(score)

    while len(ids) > 0:
        '''
        算法的过程是在一个排序好的列表，把一个概率最大值像素对应的候选框，分别与其他像素的候选框计算重叠比，达到iou阈值就把这些候选框drop掉
        把最大概率像素的下标放进去储存下标的list，迭代，直到排序好的列表为空
        '''
        # 概率最大值像素对应的候选框
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # 重叠区域的左上角坐标
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])
        # 重叠区域的右下角坐标
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])
        # 重叠面积的宽和高
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)
        # 计算重叠面积
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # iou计算
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # 删除排序列表中重叠比大于概率最大的下标
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick

def adjust_box(bboxes, offsets):
    """用于修正，微调特征图对应的候选框在实际图中的位置


    Arguments:
        bboxes: a float numpy array of shape [n, 5].
                每一个特征对图应像素的候选框坐标和概率
        offsets: a float numpy array of shape [n, 4].
                 网络回归得到的人脸边框的回归值偏差

    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # 下面的计算过程：因为bounding box回归算法的就是通过回归问题找出相对于gt偏移量，所以通过偏移量校正回原来图片，就得到了线性转移的边框坐标
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h

    translation = np.hstack([w, h, w, h])*offsets #每一个像素预测坐标与真实的车牌坐标对应的偏差
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def fix_bboxes(bboxes, width, height):
    """
    裁剪太大候选框并获得裁剪的坐标，这个函数的目的是将那些坐标的值大于或者小于原图尺寸的候选框的坐标值修改为对应原图位置的边界坐标
    有些候选框出图了，要修补这些候选框，使候选框在图中准确覆盖而不出界。

    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            每一行 (xmin, ymin, xmax, ymax, score).score是像素的概率
        width: a float number.
        height: a float number.

    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            校正之后的边框的坐标，左上角坐标以及长宽.
        y, x, ey, ex: a int numpy arrays of shape [n],
            候选框的 ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            候选框的长宽.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    x2=np.clip(x2, x1, None)
    y2=np.clip(y2, y1, None)
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' 代表是结束
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # 因为要从图中把车牌的区域切割袭来
    # (x, y, ex, ey) 是要在原图中校正过后候选框的坐标

    # (dx, dy, edx, edy) 从切割后图中经过对齐之后的候选框

    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # 出界情况，边框右边出界，太右
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    #出界情况，边框右边出界，太低
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0


    #出界情况，边框左边边出界，太左
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    #出界情况，边框左边边出界，太高
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list

def IoU(box, boxes):
    """计算两个区域的IOU

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    # box = (x1, y1, x2, y2)
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)


    #获得预测的box和gt的交集的offset
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 计算边框的宽和高
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr