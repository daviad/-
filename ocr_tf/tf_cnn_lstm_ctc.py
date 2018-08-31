# -*- coding: utf-8 -*-
from genIDCard import *
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 定义一些常量
# 图片大小，32 x 256
OUTPUT_SHAPE = (32, 256)
# 训练最大轮次
num_epochs = 10000

num_hidden = 64
num_layers = 1

obj = gen_id_card()

num_classes = obj.len + 1 + 1  # 10位数字 + blank + ctc blank

# 初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

DIGIST = '0123456789'
BATCHES = 10
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE



