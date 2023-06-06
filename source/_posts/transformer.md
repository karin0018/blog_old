---
title: Attention Is All You Need
toc: true
comments: true
math: true
date: 2021-11-19 23:09:55
tags: 论文阅读
categories: 跟李沐学 AI
---

本文为 [transformer模型原论文](https://arxiv.org/abs/1706.03762)的学习笔记。

## 预备知识

**CNN，RNN，LSTM都是什么？**

[参考文章](https://cloud.tencent.com/developer/article/1523622)

<!-- more -->
#### **卷积神经网络（Convolutional Neural Network, CNN）**

CNN 是一种前馈神经网络，通常由一个或多个卷积层（Convolutional Layer）和全连接层（Fully Connected Layer，对应经典的 NN）组成，此外也会包括池化层（Pooling Layer）。

CNN 的结构使得它易于利用输入数据的二维结构。

> 注意：前馈神经网络（Feedforward NN）指每个神经元只与前一层的神经元相连，数据从前向后单向传播的 NN。其内部结构不会形成有向环（对比后面要讲到的 RNN/LSTM）。 它是最早被发明的简单 NN 类型，前面讲到的 NN、DNN 都是前馈神经网络。

每个卷积层由若干卷积单元组成——可以想象成经典 NN 的神经元，只不过激活函数变成了卷积运算。

卷积运算是有其严格的数学定义的。不过在 CNN 的应用中，卷积运算的形式是数学中卷积定义的一个特例，它的目的是提取输入的不同特征。

一般情况下，从直观角度来看，CNN 的卷积运算，就是下图这样：


{% asset_img A1.gif This is an CNN image %}

> 上图中左侧的蓝色大矩阵表示输入数据，在蓝色大矩阵上不断运动的绿色小矩阵叫做卷积核，每次卷积核运动到一个位置，它的每个元素就与其覆盖的输入数据对应元素相乘求积，然后再将整个卷积核内求积的结果累加，结果填注到右侧红色小矩阵中。 卷积核横向每次平移一列，纵向每次平移一行。最后将输入数据矩阵完全覆盖后，生成完整的红色小矩阵就是卷积运算的结果。



CNN 结构相对简单，可以使用反向传播算法进行训练，这使它成为了一种颇具吸引力的深度学习网络模型。

{% asset_img A2.jpg This is an backword image %})

除了图像处理，CNN 也会被应用到语音、文本处理等其他领域。

#### **循环神经网（Recurrent Neural Network，RNN）**

RNN，循环神经网络，也有人将它翻译为**递归神经网络**。从这个名字就可以想到，它的结构中存在着“环”。

确实，RNN 和 NN/DNN 的数据单一方向传递不同。RNN 的神经元接受的输入除了“前辈”的输出，还有自身的状态信息，其状态信息在网络中循环传递。

RNN 的结构用图形勾画出来，是下图这样的：

{% asset_img A3.jpg This is an RNN image %}

图1

> 注意：图中的 AA 并不是一个神经元，而是一个神经网络块，可以简单理解为神经网络的一个隐层。

RNN 的这种结构，使得它很适合应用于序列数据的处理，比如文本、语音、视频等。这类数据的样本间存在顺序关系（往往是时序关系），每个样本和它之前的样本存在关联。

RNN 把所处理的数据序列视作时间序列，在每一个时刻 $t$，每个 RNN 的神经元接受两个输入：当前时刻的输入样本 $x_t$，和上一时刻自身的输出 $h_{t-1}$

$t 时刻输出：h_t=F_{\theta}(h_{t-1},x_t)$

图1经过进一步简化，将隐层的自连接重叠，就成了下图：

{% asset_img A4.jpg This is an siplifiedRNN image %}

图2

上图展示的是最简单的 RNN 结构，此外 RNN 还存在着很多变种，比如双向 RNN（Bidirectional RNN），深度双向 RNN（Deep Bidirectional RNN）等。

RNN 的作用最早体现在手写识别上，后来在语音和文本处理中也做出了巨大的贡献，近年来也不乏将其应用于图像处理的尝试。

#### **长短时记忆（Long Short Term Memory，LSTM）**

LSTM 可以被简单理解为是一种神经元更加复杂的 RNN，处理时间序列中当间隔和延迟较长时，LSTM 通常比 RNN 效果好。

相较于构造简单的 RNN 神经元，LSTM 的神经元要复杂得多，每个神经元接受的输入除了当前时刻样本输入，上一个时刻的输出，还有一个元胞状态（Cell State），LSTM 神经元结构请参见下图：

{% asset_img LSTM.jpg This is an CNN image %}

LSTM 神经元中有三个门：遗忘门，输入门和输出门。

遗忘门（Forget Gate)：接受 xt 和  ht-1 为输入，输出一个 0 到 1 之间的值，用于决定在多大程度上保留上一个时刻的元胞状态 ct-1。1表示全保留，0表示全放弃。

输入门（Input Gate）: 用于决定将哪些信息存储在这个时刻的元胞状态 ct中。

输出门（Output Gate）：用于决定输出哪些信息。



![lstm-a]({% asset_img LSTM-A.jpg This is an LSTM-A image %}/LSTM-A.jpg)

【LSTM 结构图】

{% asset_img RNN-A.jpg This is an RNN-A.jpg image %}

【RNN 结构图】

> 注意：如果把 LSTM 的遗忘门强行置0，输入门置1，输出门置1，则 LSTM 就变成了标准 RNN。

可见 LSTM 比 RNN 复杂得多，要训练的参数也多得多。

但是，LSTM 在很大程度上缓解了一个在 **RNN 训练中非常突出的问题：梯度消失/爆炸（Gradient Vanishing/Exploding）**。这个问题不是 RNN 独有的，深度学习模型都有可能遇到，但是对于 RNN 而言，特别严重。

梯度消失和梯度爆炸虽然表现出来的结果正好相反，但出现的原因却是一样的。

因为神经网络的训练中用到反向传播算法，而这个算法是基于梯度下降的——在目标的负梯度方向上对参数进行调整。如此一来就要对激活函数求梯度。

又因为 RNN 存在循环结构，因此激活函数的梯度会乘上多次，这就导致：

- 如果梯度小于1，那么随着层数增多，梯度更新信息将会以指数形式衰减，即发生了梯度消失（Gradient Vanishing）；
- 如果梯度大于1，那么随着层数增多，梯度更新将以指数形式膨胀，即发生梯度爆炸（Gradient Exploding）。

因为三个门，尤其是遗忘门的存在，LSTM 在训练时能够控制梯度的收敛性，从而梯度消失/爆炸的问题得以缓解，同时也能够保持长期的记忆性。

果然，LSTM 在语音处理、机器翻译、图像说明、手写生成、图像生成等领域都表现出了不俗的战绩。



## Abstract

段落大意：

主流的序列转录模型主要依赖于循环或者卷积神经网络，它们一般使用 encoder-decoder 架构。在性能最好的这些模型中，通常会在 encoder 和 decoder 之间使用注意力机制。我们提出了一个新的简单的网络架构，Transformer，它仅依赖于注意力机制，而没有用循环或者卷积。通过两个机器翻译实验的验证，证明 Transformer 能比现有的模型效果更好且训练时间更快，最后写明 Transformer 也能很好的泛化到其他的任务上。

总结：

前人工作+本篇文章的创新点+模型的性能很棒



## Conclusion

段落大意：

我们首次提出了一个仅依赖于注意力机制的模型，Transformer。它用 multi-head self-attention 层替换了前人用的循环层。

在机器翻译的任务上，Transformer 起到了很好的效果。

基于纯注意力机制的模型在机器翻译任务上取得的优越效果令人激动，我们认为他还能适用在图片/语音/视频等材料的研究任务中，making generation less sequential 也是一个新的研究目标。

总结：

主要贡献是提出了仅依赖于注意力机制的模型，重点在 multi-head self-attention 层。

展望未来：这种方法还能泛化到其他任务中。



## Introduction

段落大意：

> 基本是 Abstract 的扩充

