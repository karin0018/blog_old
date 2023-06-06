---
title:  Incorporating BERT into Parallel Sequence Decoding with Adapters
toc: true
comments: true
math: true
date: 2022-01-04 16:44:38
tags: 读论文
categories: NMT
---

Author:Junliang Guo, Zhirui Zhang, Linli Xu, Hao-Ran Wei, Boxing Chen, Enhong Chen
[paper](https://arxiv.org/abs/2010.06138)

## Introduction

预训练模型 bert 在自然语言处理任务上应用广泛，譬如阅读理解，文本分类等。但是目前还不能很好的应用在基于 seq2seq 框架的神经网络机器翻译（NMT）。

本文认为主要存在下列三个问题：

1. encoder 端 - 灾难性遗忘：如果用 bert 来初始化 encoder 的输入再fine-tuning（Incorporating BERT into Neural Machine Translation），翻译效果并不好。模型会忘记老数据的表征。

2. decoder 端 - bert 是条件独立的模型，但是 NMT 是非条件独立的。bert 用非自回归编码，NMT 用自回归编码。

3. 预训练模型的参数规模过大，微调不适用于小规模样本，它对学习率和模型参数不鲁棒。

   

<!--more-->

分别用下列三种方法解决：

1. encoder 端：在Bert 的每层都插入 adapter 层（轻量级神经网络模块）映射 bert 的语义空间，bert 自身的参数不做改变。

2. decoder 端：encoder 的输出作为 decoder 的输入，用 mask-predict 的非自回归算法进行训练和解码。

   这样我们的 decoder 就变得 conditional and Non-autoregressive

3. 框架中的 adapter 层都是轻量级的 MLP，参数规模小，对学习率和模型参数规模鲁棒。

4.  Each component  in the framework can be considered as a plug-in unit, making the framework very flexibleand task agnostic.



## Background

1. pre-trained language model : bert 的预训练任务之一 - MLM 掩码语言模型

2. 将预训练模型用于 seq2seq 的架构来做机器翻译（文本生成）：前人都是在一端用 bert ，我们用两端（？）【迷惑，不是也有用两端的吗】是说没有完全用 bert？

   > Knowledge distillation [15, 18] is applied in [37, 38, 4] to transfer the knowledge from BERT toeither the encoder [38] or decoder side [37, 4]. Zhu et al. [41] introduces extra attention basedmodules to fuse the BERT representation with the encoder representation

3. fine-tuning with adapters ：

   微调是指：基于预训练模型提供的语义表征（bert 最后一层的隐藏层输出），根据下游任务的特性进行领域适配，使之与下游任务的形式更加契合，获得更好的应用效果。

   适配器：通常是加入到预训练模型的内层以实现对下游任务的自适应的轻量级神经网络。

   我们探索在适配器的帮助下将来自不同领域的两个预先训练的模型融合进一个序列到序列的框架

4. 并行解码：conditional mask language model（CMLM）

   训练损失函数：

   $$
   L_{CMLM}(y^m|y^r,x;\theta_{enc},\theta_{dec}) = -\sum_{t=1}^{|y^m|}logP(y_t^m|y^r,x;\theta_{enc},\theta_{dec})\\
   loss = L_{CMLM}  +L_{length}
   $$

   | train\test | train  | train                   | test   | test                    |
   | ---------- | ------ | ----------------------- | ------ | ----------------------- |
   | **AT/NAT** | **AT** | **NAT**                 | **AT** | **NAT**                 |
   | input      | $y$    | $z = f(x;\theta_{enc})$ | $y<t$  | $z = f(x;\theta_{enc})$ |
   | output     | $y$    | $y$                     | $y$    | $y$                     |

   

   ## Framework

   {% asset_img framework.png framework  %}

   

   训练流程：

   - x 给 encoder

   - 利用 encoder 的 output 来预测 y 的长度 length

   - 初始化 decoder 的输入，是长度为 length 的 mask 序列

   - decoder 预测，得到预测单词的置信度（probability），选择最差的 k 个词再次 mask 住，作为新一轮 decoder 的输入

     $k = |y|(T-t)/T$

   - 达到迭代上限值（超参数），或者两次 decoder 出的结果相同时，迭代预测停止。

   ### why mask-predict?

   [Non-Autoregressive Neural Machine Translation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.02281v1)

   {% asset_img NAT.jpg NAT2018 %}

   Jiatao Gu在这篇ICLR18的文章中首次提出了Non-Autoregressive Translation （以下简称NAT）的概念，它对每个位置的词语的概率分布独立建模，测试时并行解码，但存在“多模态”问题（multi-modality problem）

   > multi-modality problem：同一句话很大概率存在不同的有效翻译
   >
   > e.g. 他中文说的很好（汉译英）
   >
   > 1. He is good at Chinese.
   > 2. He speaks Chinese very well.

   独立对单词建模就很可能导致这样的翻译结果 `He is Chinese very well. `

   **mask predict 的解决方法**：迭代优化模型输出，已知一部分信息能减少多模态中的可选项（lower prediction modality）

   > 论文指路：[Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.09324)

## Experiments 

{% asset_img table1.png table1 %}

While inference,we generate multiple translation candidates by taking the top B length predictions into consideration,and select the translation with the **highest probability** as the final result.

We set B = 4 for all tasks. And the upper bound of iterative decoding is set to 10。

For autoregressive decoding, we use beamsearch with width 5 for all tasks. We utilize BLEU scores as the evaluation metric. 

{% asset_img table2-3.png table2-3 %}

本文还针对模型的参数规模、模型组件、微调策略做了 ablation study，验证得出本文的模型对参数规模鲁棒且效果确实强于 BERT-Fuesd.


## Conclusion

- 提出了新的框架 AB-Net，在源端和目标端分别用预训练的 bert 初始化 encoder 和 decoder，通过在 bert 中插入 adapter层将预训练模型很好的用在了 seq2seq 架构中。

- 用 mask-predict 算法并行解码，以匹配 bert 的双向性和条件独立性。

- 保证 bert 原参数不变，解决灾难性遗忘问题

  

