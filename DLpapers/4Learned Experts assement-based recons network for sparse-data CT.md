# Learned Experts assement-based recons network for sparse-data CT
## Introduction
1. Compressive sensing 压缩感知，突破了Nyquist-Shannon采样定理，可以以低于原始频率2倍的采样率采样并还原
2. Hu etal  采用total variation作为稀疏转换的正则项，但是TV没法处理blocky artifacts，对其改进有：nonlocal means, tight wavelet frames, dictionary learning, low rank...
3. 基于压缩感知重建方法的缺点：a 迭代算法耗时长； b 难以找到通用的正则项； c 参数要设定的太多
4. FBP在低采样率下会失效，所以本文将迭代重建框架融合到LEARN网络中，优点：速度快，正则项参数均可自主学习，生成的图像结果很好
## Methods
### regualarized CT reconstruction
1. CT 重建目标： Ax = y， y是i维*校正和log变换后的观测数据*， A是(j ,i)投影矩阵 ，x是j维*离散衰减系数的向量（denotes a vector of discrete attenuation coefficients for a patient image）*， 重建目标就是从A和y得到未知的x
2. 稀疏数据FBP重建会得到很多伪影， 迭代重建效果更好因为它使用了先验知识。
3. 正则目标方程：  
   *x = argminx 0.5(Ax-y)（Ax-y）+ aR(x)*  
   Ax-y是求的L2范数，R(x)是正则项。
### LEARN net
