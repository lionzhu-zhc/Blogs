# A sparse-view CT reconstruction method based on combination of DenseNet and Deconvolution

## Introduction
- excessive x-ray radiation exposure which will potentially induce lifetime risk 
- the principle of ALARA (as low as reasonably achievable)
- Balancing image quality and x-ray dose level has become a well-known trade-off problem.
- Two strategies to lower x radiation dose:
1. lower exposure dose in single x-ray image, by adjusting the tube current or exposure time of an x-ray tube, wil produce *noise projection*
2. decrease the number of projections for a given scannning trajectory, insufficient projection data is inevitable, such as streaking artifacts, image distortion.
- Three categories of sparse-view CT recons:
1. sinogram completion, key idea is to complement the sparse projection before image reconstruction associated with analytical algorithm, such as FBP.
2. iterative recons: statistical properties of data in projection and prior info, *complicated para tuning* 
3. post processing: 3D filtering
## Method
### Method over-view
1. proposed an architecture: DD-Net
2. dense block: BN+Relu+Conv and then concatenate the above.
3. deconv
4. shortcut connection: add the details for deconv
### Experimental settings
- training: 2000imgs, testing: 500 imgs
## Exp Results
1. Chest CT and Hip CT
2. use FBP res as input, compare with SART-TV, NLM, BM3D
3. evaluation: peak signal-to-noise (PSNR), root-mean-square error (RMSE) and structural similarity (SSIM) 
## Conclusion
- Compared to conventional algorithms that are based on sinogram completion, DD-Net is based in image domain, and it can avoid any new artifacts. This is because any operation in projection domain will influence the global information in image domain. 
- deconv is a little better than conv
- shortcut is very useful
-  