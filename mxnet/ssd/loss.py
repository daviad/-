from mxnet.contrib.ndarray import MultiBoxTarget
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet import gluon

def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axex=(0.2,1))
    z = MultiBoxTarget(*[default_anchors, labels,class_predicts])
    box_target = z[0]  # 预设框偏移量 (x, y, width, height)
    box_mask = z[1]  # box_mask用来把负类的偏移量置零，因为背景不需要位置！
    cls_target = z[2] # 每个预设框应该对应的分类
    return box_target, box_mask, cls_target

class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss,self).__init__(None, batch_axis,**kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pt = F.pick(output, label, axis = self._axis, keepdims = True)
        loss = -self._alpha * ((1 - pt) ** self._gamma) * F.log(pt)
        return F.mean(loss, axis = self._batch_axis, exclude = True)

# cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
cls_loss = FocalLoss()
print(cls_loss)

class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss,self).__init__(None, batch_axis, **kwargs)
    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar= 1.0)
        return F.mean(loss, self._batch_axis, exclude=True)

box_loss = SmoothL1Loss()
print(box_loss)

