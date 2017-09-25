# 4-Learned Experts assement-based recons network for sparse-data CT
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
1. 在每次迭代中，x被*循环残差CNN*卷积。
2. eq.8的第三系通过Conv和RELU实现，其网络结构如下：
   ![1](https://github.com/lionzhu6336/Blogs/raw/master/DLpapers/4-eq8.PNG)
   ![1](https://github.com/lionzhu6336/Blogs/raw/master/DLpapers/4-fig1.PNG)
3. 整个LEARN网络结构：
   ![1](https://github.com/lionzhu6336/Blogs/raw/master/DLpapers/4-fig2.PNG)
4. loss fun 是重建的Xt与优质图Xs之间的MSE，用adam优化。一堆计算梯度的公式...
### Exp and results
1. 数据集：NIH-AAPM-Mayo clinic low dose CT grand challenge: 5936,512x512 CT images of 10 patients.
2. 参考图像是用所有2304个投影景FBP得到的优质图像。 下采样的数据被采到了64 或128view
3. learn 网络用成对的全剂量及采样图像训练，初始输入为FBP结果，三层卷积分别为24@3x3，24@3x3， 1@3x3， 迭代次数50
4. 量化评估指数：**RMSE, PSNR(peak signal to noise ratio), SSIM(structural similarity index mearsure)**
5. 对比方法：ASD-POCS, dual dictionary learning, FBP Convnet
#### visulization-based evaluation
1. FBPConv 会损失很多细节，原因有三：大量的卷积反卷积会丢失细节；后处理方法依靠大量的训练数据；只使用了投影数据
### quantitative eval
1. 64和128view的模型PSNR分别增强了5.7 5.3dB
2. p <0.05的标准差
### trade-off between network and performance
1. 64 view模型重建最终设定：24@3x3, iteration 50, training samples: 200

##COnclusion
In our future work, we will further optimize the LEARN network for clinical applications by training the system matrix as well and generalizing the loss function in the GAN framework.
