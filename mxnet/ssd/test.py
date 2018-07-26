import numpy as np
import cv2
from mxnet import nd
import train
from mxnet.contrib.ndarray import MultiBoxDetection
import matplotlib.pyplot as plt

data_shape = 256


def preprocess(image):
    # 调整图片大小成网络的输入
    image = cv2.resize(image, (data_shape, data_shape))
    # 转换 BGR 到 RGB
    image = image[:, :, (2, 1, 0)]
    # 减mean之前先转成float
    image = image.astype(np.float32)
    # 减 mean
    # image -= np.array([123, 117, 104])
    # 调成为 [batch-channel-height-width]
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :]
    # 转成 ndarray
    image = nd.array(image)
    return image


image = cv2.imread('img/pikachu.jpg')
x = preprocess(image)
print('x', x.shape)

# 如果有预先训练好的网络参数，可以直接加载
train.net.load_params('ssd_%d.params' % train.epochs, train.ctx)
anchors, cls_preds, box_preds = train.net(x.as_in_context(train.ctx))
print('anchors', anchors)
print('class predictions', cls_preds)
print('box delta predictions', box_preds)

# 跑一下softmax， 转成0-1的概率
cls_probs = nd.SoftmaxActivation(nd.transpose(cls_preds, (0, 2, 1)), mode='channel')
# 把偏移量加到预设框上，去掉得分很低的，跑一遍nms，得到最终的结果
output = MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress = True, clip=False)
print(output)


def display(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img)
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        # text = class_names[ic]
        text = "pikachu"
        plt.gca().text(xmin, ymin - 2,'{:s} {:.3f}'.format(text, score),bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.show()


display(image[:, :,(2, 1, 0)], output[0].asnumpy(), thresh=0.5)
print('test over!')