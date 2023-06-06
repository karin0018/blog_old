---
title: chap5 深度学习计算
toc: true
comments: true
math: true
date: 2021-12-13 16:36:01
tags: DeepLearning
categories: 跟李沐学 AI
---
[本章教材地址](https://zh.d2l.ai/chapter_deep-learning-computation/index.html), [本章课程视频](https://www.bilibili.com/video/BV1AK4y1P7vs?spm_id_from=333.999.0.0)

> 到目前为止，我们已经介绍了一些基本的机器学习概念，并慢慢介绍了功能齐全的深度学习模型。在上一章中，我们从零开始实现了多层感知机的每个组件，然后展示了如何利用高级API轻松地实现相同的模型。为了易于学习，我们调用了深度学习库，但是跳过了它们工作的细节。在本章中，我们开始深入探索深度学习计算的关键组件，即模型构建、参数访问与初始化、设计自定义层和块、将模型读写到磁盘，以及利用GPU实现显著的加速。这些知识将使你从基础用户变为高级用户。虽然本章不介绍任何新的模型或数据集，但后面的高级模型章节在很大程度上依赖于本章的知识。

本章没有介绍新的概念，教材中的代码复现在 [这里](https://github.com/karin0018/d2l_MuLi)，欢迎指正！
