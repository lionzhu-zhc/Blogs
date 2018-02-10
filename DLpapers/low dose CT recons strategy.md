# low dose CT recons strategy
## 深度滤波反投影网络
1. 网络学习内容：原始投影数据 ——> 滤波后的投影数据   
      再由fbp得到重建数据
2. 网络输入：可在空间域对 p(s, theta)作为网络输入处理；  
   也可在频率域对上面做傅立叶变换的 P(s, theta)作为输入，需要做个对数变换预处理。
3. loss计算： 以常规剂量ramp滤波后的投影数据作为标签；  
      或 以高质量CT图像作为标签。  
## 投影到图像的全网络
### 在投影空间的前处理网络
1. input：低剂量光子信号  
   output： 低剂量图像中包含的噪声  
2. 原始投影数据进行指数变换，使其信号分布范围更广， 然后通过Anscombe变换把含有泊松噪声的数据转换为高斯分布，这样更适合卷积网络
3. 图像重建是把原低剂量ct投影数据 - 网络output得到去噪投影数据，然后Anscombe逆变换，在指数变换得到最终投影数据
### 解析重建
  对前处理网络的结果投影数据，可采用ramp滤波反投影、自适应滤波反投影以及**深度滤波反投影**网络进行重建
### 图像空间后处理网络
1. 采用densenet的conv block结构， 缓解了梯度消失的问题，同时充分利用了前面的特征图
2. 采用孔洞卷积核，可以增大感受野同时减少参数
## 蛙跳式迭代重建网络
### 迭代重建
1. log变换后ct投影数据近似服从独立高斯分布
2. 迭代重建的目标函数包含数据保真项和先验项
3. **保真项**：引入信息散度刻画刻画实际数据与估计投影数据之间的误差
4. **先验项**：TV， dictionary， low rank decomposition
### 蛙跳网络
以三个CNN代替几十次的迭代重建