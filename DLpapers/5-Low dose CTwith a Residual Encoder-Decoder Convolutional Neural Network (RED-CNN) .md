# 5- Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)
## Introduction
1. most denoising models have limited layers(2-3 layers usually)
## Methods
### Key ideas:
1. The idea of the autoencoder, which was originally designed for training with noisy samples, was introduced into our model, and convolution and deconvolution layers appeared in pairs;
2. To avoid losing details, pooling layer was discarded;
3. Convolution layers can be seen as noise filters in our application, but filtering leads to loss in details. Deconvolution and shortcutting in our model were used for detail preservation
### Red CNN
1. overall structure:
   ![1](https://github.com/lionzhu-zhc/Blogs/blob/master/DLpapers/5-fig1.png)
  

### Exp and results
1. 临床数据集clinical data：NIH-AAPM-Mayo clinic low dose CT grand challenge: 5936,512x512 CT images of 10 patients.
2. 仿真数据: National Biomedical Imaging Archive (NBIA) 256x256, norm dose CT, 加Poisson噪声仿真低剂量图像
