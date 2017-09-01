# Deep Neural Network-Based Computer-Assisted Detection of Cerebral Aneurysms in MR Angiography
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; J. Magnetic Resonance Imaging 2017.05
## Task
computer-assited detection in cerebral aneurysms  
need low fp rate as well as high sensitivity
1. develop a CAD (computer-assist detection) system for intracranial aneurysms in unenhanced MRA based on CNN and maximum intensity projection(MIP)
2. demostrate the usefulness of system with large dataset

## Materials and methods
1. VOI: 16x16x16 cube, and the 16x16 imnages from 9 direction  
* Each input image is classified as “positive” when the voxel corresponding to the input image is inside an aneurysm and “negative” when the voxel is outside the aneurysm(s) *

2. train set:300, validation set: 50, test set: 100  
utilize the AUC of FROC to determine the hyperparas.

![1](https://github.com/lionzhu6336/Blogs/raw/master/DLpapers/2-1.PNG)

## Results
70% sensity at 0.26 FPs/case , 89.2s per case

## Dis
employ *MIP* to  represent 3D image as 2D

**results showed that the combination of a CNN and an MIP algorithm can significantly decrease FPs**