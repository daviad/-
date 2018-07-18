from mxnet.gluon import nn
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet import gluon


def class_predictor(num_anchors, num_class):
    return nn.Conv2D(num_anchors * (num_class + 1), 3, padding=1)


def box_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, 3, padding=1)


def down_sample(num_filters):
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, kernel_size=3, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    out.hybridize()
    return out


def flatten_prediction(pred):
    return nd.flatten(nd.transpose(pred, axex=(0, 2, 3, 1)))


def concat_predictions(preds):
    return nd.concat(*preds, dim=1)


def body():
    out = nn.HybridSequential()
    for nfilters in [16,32,64]:
        out.add(down_sample(nfilters))
    return out


def toy_ssd_model(num_anchors, num_classes):
    downsamples= nn.Sequential()
    class_preds = nn.Sequential()
    box_preds = nn.Sequential()

    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))

    for scale in range(5):
        class_preds.add(class_predictor(num_anchors, num_classes))
        box_preds.add(box_predictor(num_anchors))

    return body(), downsamples, class_preds, box_preds


# print(toy_ssd_model(5, 2))

def toy_ssd_forward(x, body, downsamples, class_preds, box_preds, sizes, ratios):
    x = body(x)
    # 在每个预测层, 计算预设框，分类概率，偏移量
    # 然后在下采样到下一层预测层，重复
    default_anchors = []
    predicted_boxes = []
    predicted_classes = []

    for i in range(5):
        default_anchors.append(MultiBoxPrior(x,sizes=size[i], ratios=ratios[i]))
        predicted_boxes.append(flatten_prediction(box_preds[i]))
        predicted_classes.append(flatten_prediction(class_preds[i]))
        if  i < 3:
            x = downsamples[i](x)
        elif i == 3:
            # 最后一层可以简单地用全局Pooling
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))
    return default_anchors,predicted_classes, predicted_boxes


class ToySSD(gluon.Block):
    def __init__(self,num_class,**kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # 5个预测层，每层负责的预设框尺寸不同，由小到大，符合网络的形状
        self.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # 每层的预设框都用 1，2，0.5作为长宽比候选
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_class

        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = toy_ssd_model(4, num_classes)

    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = toy_ssd_forward(x, self.body, self.downsamples,
                                                                              self.class_preds, self.box_preds,
                                                                              self.anchor_sizes, self.anchor_ratios)

        # 把从每个预测层输入的结果摊平并连接，以确保一一对应
        anchors = concat_predictions(default_anchors)
        box_preds = concat_predictions(predicted_boxes)
        class_preds = concat_predictions(predicted_classes)
        # 改变下形状，为了更方便地计算softmax
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))

        return anchors, class_preds, box_preds


# 新建一个2个正类的SSD网络
net = ToySSD(2)
net.initialize()
x = nd.zeros((1, 3, 256, 256))
default_anchors, class_predictions, box_predictions = net(x)
print('Outputs:', 'anchors', default_anchors.shape, 'class prediction', class_predictions.shape, 'box prediction', box_predictions.shape)


