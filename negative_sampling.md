

​		Word2vec的另一个瓶颈在于中间层之后的处理，即矩阵乘积和Softmax层的计算，使用负采样(Negative Sampling)的方法进行训练可以解决

​		相比原来需要计算所有字典中所有词的预测误差，负采样方法只需要对采样出的几个负样本计算预测误差，Word2vec模型的优化目标从多分类问题退化成了近似二分类问题



#### 中间层之后存在的问题

![1646476168083](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646476168083.png)

 	此时，在以下两个地方需要很多计算时间

- 中间层的神经元和权重矩阵($\boldsymbol W_\rm{out}$)的乘积 

  中间层向量的大小是100，权重矩阵的大小是100×1000000，如此巨大的矩阵乘积计算需要大量时间和内存

- Softmax层的计算

  随着词汇量的增加，Softmax层的计算量也会增加
  $$
  y_k=\frac{\exp(s_k)}{\sum_{i=1}^{1000000}\exp(s_i)}
  $$



#### 从多分类到二分类

​		负采样方法的关键思想在于用二分类拟合多分类

- 多分类：当上下文是“you”和“goodbye”时，目标词是什么？
- 二分类：当上下文是“you”和“goodbye”时，目标词是“say”吗？

​		使用二分类时，输出层只需要一个神经元，即单词"say"的得分![1646476951705](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646476951705.png)

​		因为输出层的神经元只有一个，因此要计算中间层和我输出层的权重矩阵的乘积，只需要提取单词"say"对应的列，并用它与中间层的神经元计算内积即可

![1646477483254](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646477483254.png)

​		输出侧的权重$\boldsymbol{W_\rm {out}}$中保存了各个单词ID对应的单词向量，我们提取"say"这个单词向量，再求这个向量和中间层神经元的内积，就是最终的得分。



#### sigmoid函数和交叉熵误差

​		要使用神经网络解决二分类问题，需要使用 sigmoid 函数将得分转化为概率

​		为了求损失，我们使用交叉熵误差作为损失函数

> 在多分类的情况下，输出层使用Softmax函数将得分转化为概率，损失函数使用交叉熵误差
>
> 在二分类的情况下，输出层使用sigmoid 函数，损失函数也使用交叉熵误差

​		![1646477693202](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646477693202.png)
$$
y=\frac{1}{1+\exp(-x)}
$$

$$
\frac{dy}{dx}=-\frac{1}{(1+\exp(-x))^2}\cdot(-1)\cdot\exp(-x)=y(1-y)
$$

​		通过 sigmoid 函数得到概率 y 后，可以由概率y计算损失
$$
L=-(t\log y+(1-t)\log(1-y))
$$
​		其中，y是sigmoid函数的输出，t是正确解标签，取值为1时表示正确解是“Yes”，输出$-\log y$，取值为0时表示正确解是“No”，输出$-\log(1-y)$

![1646478232551](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646478232551.png)



#### 负采样

​		目前我们只学习了正确答案，还不确定错误答案会有怎么样的结果

​		在之前的例子中，上下文是“you”和“goodbye”，目标词是“say”，如果此时模型有好的权重，则Sigmoid层的输出将接近1

​	![1646478401853](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646478401853.png)

​		而我们真正要做的是，对于正确答案“say”，输出概率接近1，对于其他单词（错误答案），输出概率接近0

![1646478451972](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646478451972.png)

​		因此我们还需要继续学习错误答案，但并不是所有的错误答案，**只使用少数负例**，将正例和部分负例的损失加起来，作为最终的损失![1646478604531](C:\Users\12436\AppData\Roaming\Typora\typora-user-images\1646478604531.png)

​		正例”say“和之前一样，向Sigmoid with Loss层输入解标签1，负例“hello”和”i“是错误答案，向Sigmoid with Loss层输入解标签0，将各个数据的损失相加，作为最终损失输出
$$
E=-\log\sigma(\boldsymbol{{\nu^{'}_{w_o}}^Th})-\sum_\rm{w_j\in \boldsymbol W_\rm {neg}}\log\sigma(-\boldsymbol{{\nu^{'}_{w_j}}^Th})
$$
$$
1-\frac{1}{1+\exp(-x)}=\frac{\exp(-x)}{1+\exp(-x)}=\frac{1}{1+\exp(x)}
$$

$\boldsymbol{\nu^{'}_{w_0}}$是输出词向量（正样本），$\boldsymbol h$是隐层向量，$\boldsymbol W_\rm {neg}$是负样本集合，$\boldsymbol{\nu^{'}_{w_0}}$是负样本词向量

