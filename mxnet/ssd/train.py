import loadData
import net_ssd
import loss as myLoss
import mxnet as mx
from mxnet import nd
from mxnet import gluon
import os.path as osp
from mxnet.gluon import utils as gutils

import time
from mxnet import autograd as ag


ctx = mx.gpu() # 用GPU加速训练过程
try:
    _ = nd.zeros(1, ctx=ctx)
    # 为了更有效率，cuda实现需要少量的填充，不影响结果
    loadData.train_data.reshape(label_shape=(3, 5))
    loadData.train_data.sync_label_shape(loadData.train_data)
except mx.base.MXNetError as err:
    # 没有gpu也没关系，交给cpu慢慢跑
    print('No GPU enabled, fall back to CPU, sit back and be patient...')
    ctx = mx.cpu()


net = net_ssd.ToySSD(loadData.num_class)
net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})


epochs = 150
log_interval = 20
from_scratch = False
if from_scratch:
    start_epoch = 0
else:
    start_epoch = 148
    pretrained = 'ssd_pretrained.params'
    shal = 'fbb7d872d76355fff1790d864c2238decdb452bc'
    url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/ssd_pikachu-fbb7d872.params'
    if not osp.exists(pretrained):
        print('downloading', pretrained, url)
        gutils.download(url,pretrained,shal)
    net.load_params(pretrained, ctx)


# for epoch in range(start_epoch, epochs):
#     # 重置iterator和时间戳
#     loadData.train_data.reset()
#     myLoss.cls_metric.reset()
#     myLoss.box_metric.reset()
#     tic = time.time()
#     # 迭代每一个批次
#     for i, batch in enumerate(loadData.train_data):
#         btic = time.time()
#         # 用autograd.record记录需要计算的梯度
#         with ag.record():
#             x = batch.data[0].as_in_context(ctx)
#             y = batch.label[0].as_in_context(ctx)
#             default_anchors, class_predictions, box_predictions = net(x)
#             box_target, box_mask, cls_target = myLoss.training_targets(default_anchors, class_predictions, y)
#             # 损失函数计算
#             loss1 = myLoss.cls_loss(class_predictions, cls_target)
#             loss2 = myLoss.box_loss(box_predictions, box_target, box_mask)
#             # 1比1叠加两个损失函数，也可以加权重
#             loss = loss1 + loss2
#             # 反向推导
#             loss.backward()
#             # 用trainer更新网络参数
#         trainer.step(loadData.batch_size)
#         # 更新下衡量的指标
#         myLoss.cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
#         myLoss.box_metric.update([box_target], [box_predictions * box_mask])
#         if (i + 1) % log_interval == 0:
#             name1, val1 = myLoss.cls_metric.get()
#             name2, val2 = myLoss.box_metric.get()
#             print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
#                   % (epoch, i, loadData.batch_size / (time.time() - btic), name1, val1, name2, val2))
#
#         # 打印整个epoch的的指标
#     name1, val1 = myLoss.cls_metric.get()
#     name2, val2 = myLoss.box_metric.get()
#     print('[Epoch %d] training: %s=%f, %s=%f' % (epoch, name1, val1, name2, val2))
#     print('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))
#
# # 还可以把网络的参数存下来以便下次再用
# net.save_params('ssd_%d.params' % epochs)


