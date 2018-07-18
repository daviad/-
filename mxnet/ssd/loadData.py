import mxnet.image as image
import os

# edge_size：输出图片的宽和高。
def load_data_pikachu(batch_size, edge_size=256):
    data_dir = './data/pikachu'
    train_iter = image.ImageDetIter(
        path_imgrec =os.path.join(data_dir, 'train.rec'),
        # 每张图片在 rec 中的位置，使用随机顺序时需要。
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), # 输出图片形状。
        shuffle=True, # 用随机顺序访问。
        rand_crop=1, # 一定使用随机剪裁。
        min_object_covered=0.95, # 剪裁出的图片至少覆盖每个物体 95% 的区域。
        max_attempts=200) # 最多尝试 200 次随机剪裁。如果失败则不进行剪裁。
    val_iter = image.ImageDetIter( # 测试图片则去除了随机访问和随机剪裁。
        path_imgrec=os.path.join(data_dir, 'val.rec'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=False)
    return train_iter, val_iter

batch_size = 32
edge_size = 256
train_iter, _ = load_data_pikachu(batch_size, edge_size)
batch = train_iter.next()
# print(batch.data[0].shape, batch.label[0].shape)
print(batch)
# def get_iterators(data_shape,batch_size):
    # class_name = ['picachu']
    # num_class = len(class_name)
    # train_iter = image.ImageDetIter(
    #     batch_size=batch_size,
    #     data_shape=(3,data_shape,data_shape),
    #     path_imgrec='./data/pikachu/train.rec',
    #     part_index='./data/pickachu/train.idx',
    #     shuffle=True,
    #     # mean=True,
    #     # rand_crop=1,
    #     # min_object_covered=0.95,
    #     # max_atttempts=200
    #     )
    # val_iter = image.ImageDetIter(
    #     batch_size=batch_size,
    #     data_shape=(3,data_shape,data_shape),
    #     path_imgrec='./data/pikachu/val.rec',
    #     shuffle=False,
    #     # mean=True
    # )
    # return train_iter,val_iter,class_name,num_class


# train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)


import numpy as np
import matplotlib.pyplot as plt

img = batch.data[0][0].asnumpy()  # 取第一批数据中的第一张，转成numpy
img = img.transpose((1,2,0))  # 交换下通道的顺序
# img += np.array([123,117,104])
img = img.astype(np.uint8)  # 图片应该用0-255的范围
# 在图上画出真实标签的方框
for label in batch.label[0][0].asnumpy():
    if label[0] < 0:
        break
    print(label)
    xmin, ymin, xmax, ymax = [int(x *edge_size) for x in label[1:5]]
    rect = plt.Rectangle((xmin,ymin),xmax - xmin, ymax - ymin, fill=False,edgecolor=(1,0,0),linewidth=3)
    plt.gca().add_patch(rect)
plt.imshow(img)
plt.show()
