# Automated Melanoma Recognition in dermoscopy images via very deep residual networks
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; IEEE TMI 2017.04
## Introduction
### Challenge:
1. color, texture,shape, size and location in the dermoscopy images as well as the high degree of visual similarity between melanoma and non-melanoma lesions 
2. low contrasts and obscure boundaries between skin lesions
3. presence of artifacts 

### Existing methods:
1. low-level hand-crafted features, select features and combine features
> these hand-crafted features are incapable of dealing with the huge intraclass variation of melanoma and the high degree of visual similarity between melanoma and non-melanoma lesions
2. segmentation first and then recognize melanomas
3. intergrate CNN, sparse coding, SVM to recognization
4. FCN based on Alexnet

### Medical Images with CNN difficulties:
1. limited training data to train very deep CNN with lots of paras
2. interclass (类内) varaition is small 

### Main contributions:
1. so deep 50+ lyaers, 
2. deep fully conv residual network --very general
>further enhance its capability by incorporating a multi-scale contextual information integration scheme
3. compared networks with different depths

## Methods:
> We first construct a very deep fully convolutional residual network, which incorporates multi-scale feature representations, to segment skin lesions. Based on the segmentation results, we employ a very deep residual network to precisely distinguish melanomas from non-melanoma lesions.

![1](https://github.com/lionzhu6336/Blogs/raw/master/DLpapers/1-1.PNG)

1. DRN: residual net for degradation  and batch normalization for vanishing gradients
2. FCRN: Fully conv residual network ->multi-scale contextual info
         corp the original images to compass the lesion (250X250)
3. seg the lesion first and then classify
4. avg the results of softmax and SVM to obtain the finals

## Results
- dataset: Skin Lesion Analysis Towards Melanoma Detection released with ISBI 2016
- batchsize:4 momentum:0.9 weight decay: 0.0005 lr: 0.001
- pretrained on ImageNet
- *50* layers FCRN*8* has the best results
