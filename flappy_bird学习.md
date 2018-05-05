# flappy_bird学习
网上看了这个例子，写的很好，在学习过程中也遇到不少问题。现在把知识点和自己的解决方法记录下来。
[Tensorflow基于Deep Q Learning DQN 玩Flappy Bird](https://blog.csdn.net/songrotek/article/details/50951537)

## sys.path.append()
下载代码后，我直接在 sublime Text 中cmd+B 运行代码。结果总是报错：文件无法打开。没有办法阅读原代码发现这个。功能是添加系统变量。 我就在控制台中运行代码成功。这个是 sublime Text 的编译系统的问题。具体是什么原因，以后再看看 sublime tex 的官网有没有介绍。

## pygame的基础学习 
可以参考[用Python和Pygame写游戏-从入门到精通（目录）](http://eyehere.net/2011/python-pygame-novice-professional-index/) 简单易懂  

## DQN 
可以参考莫烦视频，和文中的链接。 [这个链接也还可以](http://www.tbqw.com/art/165064.html)        
这部分以前看过，但并没有完全理解。基本忘记了。这次，再学习一次，并记录下来。争取这次能理解掌握。   
### Q-learning
* experience replay

  > Deepmind使用了experience replay的技巧。简单的说就是建立一个经验池，把每次的经验都存起来，要训练的时候就 随机 的拿出一个样本来训练。这样就可以解决状态state相关的问题。以此同时，动作的选择采用常规的ϵ
-greedy policy。 就是小概率选择随机动作，大概率选择最优动作。然后呢输入的历史数据不可能是随机长度，这里就采用固定长度的历史数据，比如deepmind使用的4帧图像作为一个状态输入。



Q-learning 这套算法中一些参数的意义. Epsilon greedy 是用在决策上的一种策略, 比如 epsilon = 0.9 时, 就说明有90% 的情况我会按照 Q 表的最优值选择行为, 10% 的时间使用随机选行为. alpha是学习率, 来决定这次的误差有多少是要被学习的, alpha是一个小于1 的数. gamma 是对未来 reward 的衰减值. 


#### 关于Q 提到Q-learning，我们需要先了解Q的含义。[知乎]（https://www.zhihu.com/question/26408259）Q-learning
函数Q(s,a)是代表机器人在状态s时采取动作a所得到的未来价值回报。  
Q为动作效用函数（action-utility function），用于评价在特定状态下采取某个动作的优劣，可以将之理解为智能体（Agent，我们聪明的小鸟）的大脑。我们可以把Q当做是一张表。表中的每一行是一个状态，每一列（这个问题中共有两列）表示一个动作（飞与不飞）。这张表一共  行，表示  个状态，每个状态所对应的动作都有一个效用值。训练之后的小鸟在某个位置处飞与不飞的决策就是通过这张表确定的。小鸟会先去根据当前所在位置查找到对应的行，然后再比较两列的值（飞与不飞）的大小，选择值较大的动作作为当前帧的动作。  

 那么这个Q是怎么训练得来的呢，贴一段伪代码。
```
Initialize Q arbitrarily //随机初始化Q值
Repeat (for each episode): //每一次游戏，从小鸟出生到死亡是一个episode
    Initialize S //小鸟刚开始飞，S为初始位置的状态
    Repeat (for each step of episode):
        根据当前Q和位置S，使用一种策略，得到动作A //这个策略可以是ε-greedy等
        做了动作A，小鸟到达新的位置S'，并获得奖励R //奖励可以是1，50或者-1000
        Q(S,A) ← (1-α)*Q(S,A) + α*[R + γ*maxQ(S',a)] //在Q中更新S
        S ← S'
    until S is terminal //即到小鸟死亡为止
 ```
其中有两个值得注意的地方    
1. “根据当前Q和位置S，使用一种策略，得到动作A，这个策略可以是ε-greedy等。”这里便是题主所疑惑的问题，如何在探索与经验之间平衡？假如我们的小鸟在训练过程中，每次都采取当前状态效用值最大的动作，那会不会有更好的选择一直没有被探索到？小鸟一直会被桎梏在以往的经验之中。而假若小鸟在这里每次随机选取一个动作，会不会因为探索了太多无用的状态而导致收敛缓慢？   
于是就有人提出了ε-greedy方法，即每个状态有ε的概率进行探索（即随机选取飞或不飞），而剩下的1-ε的概率则进行开发（选取当前状态下效用值较大的那个动作）。ε一般取值较小，0.01即可。当然除了ε-greedy方法还有一些效果更好的方法，不过可能复杂很多。以此也可以看出，Q-learning并非每次迭代都沿当前Q值最高的路径前进。
2. `Q(S,A) ← (1-α)*Q(S,A) + α*[R + γ*maxQ(S',a)]` 这个就是Q-learning的训练公式了。其中α为学习速率（learning rate），γ为折扣因子（discount factor）。根据公式可以看出，学习速率α越大，保留之前训练的效果就越少。折扣因子γ越大，所起到的作用就越大。但指什么呢？小鸟在对状态进行更新时，会考虑到眼前利益（R），和记忆中的利益（）。指的便是记忆中的利益。它是指小鸟记忆里下一个状态的动作中效用值的最大值。如果小鸟之前在下一个状态的某个动作上吃过甜头（选择了某个动作之后获得了50的奖赏），那么它就更希望提早地得知这个消息，以便下回在状态可以通过选择正确的动作继续进入这个吃甜头的状态。可以看出，γ越大，小鸟就会越重视以往经验，越小，小鸟只重视眼前利益（R）。根据上面的伪代码，就可以写出Q-learning的代码了。


#### Q-learning更新公式：
```
q_target = r + self.gamma * self.q_table.ix[s_, :].max()   
self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  
```  
我看莫烦的代码实现时这样写的，不明白为啥和上面的公式不一样。最后发现时一样的。只是作了变形。 
在莫烦的视频中，把Q表中的的值叫Q估计，把 `R + γ*maxQ(S',a)` 叫Q现实

学习莫烦的第一个简单的代码例子。的理解  
对于当前的一次迭代 
Q表的估计≈Q表的假设≈Q表的初始值
Q表的现实≈通过计算得到的Q表的值

这样就有现实值估计值。类似神经网络的预测值（计算值）和标签。这样就可以优化，迭代，反向传播。。。。。。
Sarsa   与Q-learning  的区别仅仅在于，  更新Q-table 表里的某一项的时候， 是先走， 还是先计算更新而已。 没有大的区别。

再说 DQN ：
DQN   在原先的Q-learning 上做了几个处理：
1. 在选择Action 的时候， 不是用    values.max;  而是用  predict（）.max
2. 在更新的时候， 不是更新 Q-learning 里的值， 而是通过训练 定量的数据minbatch , 来更新网络的 weights 。  
更新了 weights ， 其实就是变相更新 values.max 的计算方式； 也就确定了  Action 的选择。

DQN有不少实现方式。[DQN起源，原理，核心理解](https://blog.csdn.net/Charel_CHEN/article/details/77408050)
GD(梯度下降)和SGD(随机梯度下降)

### DQN 要点
* 需要两个神经网络 （冻结神经网络）fix target network
* eva_net两个输入：
	* 状态（小鸟当前的位置等） 经过网络计算 得到 Q估计
	* Q_target（Q现实）
n_features：observation 个数 比如：长，宽 ,x,y 坐标 
b的个数和w的列数相同    n_l1 第一层有多少个神经元

#### NIPS 2013 DQN算法
1. 初始化replay memory D 容量为N
2. 用一个深度神经网络作为Q值网络，初始化权重参数
3. 设定游戏片段总数M
4. 初始化网络输入，大小为84*84*4，并且计算网络输出
5. 以概率ϵ 随机选择动作at 或者通过网络输出的Q（max）值选择动作at
6. 得到执行at后的奖励rt和下一个网络的输入
7. 根据当前的值计算下一时刻网络的输出
8. 将四个参数作为此刻的状态一起存入到D中（D中存放着N个时刻的状态）
9. 随机从D中取出minibatch个状态
10. 计算每一个状态的目标值（通过执行at后的reward来更新Q值作为目标值）
11. 通过SGD更新weight



#### NIPS 2015 DQN算法
1. 初始化replay memory D，容量是N 用于存储训练的样本
2. 初始化action-value function的Q卷积神经网络 ，随机初始化权重参数θ
3. 初始化 target action-value function的Q̂卷积神经网络，结构以及初始化权重θ和Q相同
4. 设定游戏片段总数M
5. 初始化网络输入，大小为84*84*4，并且计算网络输出
6. 根据概率 ϵ（很小）选择一个随机的动作或者根据当前的状态输入到当前的网络中 （用了一次CNN）计算出每个动作的Q值，选择Q值最大的一个动作（最优动作）

7. 得到执行at后的奖励rt和下一个网络的输入
8. 将四个参数作为此刻的状态一起存入到D中（D中存放着N个时刻的状态）
9. 随机从D中取出minibatch个状态
10. 计算每一个状态的目标值（通过执行at后的reward来更新Q值作为目标值）
11. 通过SGD更新weight
12. 每C次迭代后更新target action-value function网络的参数为当前action-value function的参数

再一次学习明白了很多,感觉这次懂了 Q-learnning 。在网上看到这一系列的文章。感觉在有了一点基础后，看这种系列文章更清晰，更能把握重点  [DQN 从入门到放弃1 DQN与增强学习](https://zhuanlan.zhihu.com/p/21262246?refer=intelligentunit)


### 强化学习RL
* 通过价值选行为（Q learning，Sarsa，Deep Q Network）
* 直接选行为 （Policy Grandients）
* 想象环境并从中学习 （Model Based RL）

## 代码分析
实现原理分析
### pygame


(python中numpy-choice函数)[https://blog.csdn.net/IAMoldpan/article/details/78707140]
