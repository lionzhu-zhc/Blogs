1.  `torch.save(net.state_dict(), 'net.pkl')` 只保存参数会更快   
    恢复时需要先再写一遍网络结构然后用 `net2.load_state_dict(torch.load(net.pkl))`
    
2.  ```
     device = torch.device('cuda:0' if torch.cuda.is_availabel() else 'cpu')   
     data = data.to(device)   
      model.to(device)
    ```
      这是把模型和数据都送进gpu加速训练
    
3.  `optimizer.zero_grad()` 当不把每次的梯度设为0时，梯度会叠加，相当于增大了batchsize   
     zero_grad 是将所有Variable的梯度清0，*不是weights*
    
4.  pytorch反向传播的计算例子，很详细：   

    > https://zhuanlan.zhihu.com/p/36294441
    
5.  `nn.Xxx`封装性更高，不需要手动设置weights bias   
     `nn.functional.Xxx` 需要手动设置w,b，且不能用在`nn.Sequential()`里面   
     建议通常情况下使用类，特殊需要使用`functional`
    
6.  `torch.max(output, 1)[1]`   
     max返回的是[max_probability, index]，1 是沿的轴向
     [1] 指的是取返回list的第二项，即index也就是label   
    
7.  `x.cuda()` 将Tensor x放到GPU上   
     `var.data.cpu().numpy()` 网络的变量需要先移动到cpu才能变成numpy，否则出错。
     或者如下：   
     ```
     device = torch.device("cuda" if use_cuda else "cpu")
     model = MyRNN().to(device)
     ```

8. `x.requires_grad_()` 可以将tensor x的梯度属性设为True

9.  ``` 
     class net(nn.Module):
     	def __init__(self, in_channel, out_channel):
          	super(net, self).__init__()
    ```
     指的是子类net调用了父类nn.Module的初始化方法，super中的`(net,self)`是python2的用法，python3中为空。
    
10.  可微IOU loss:
    ```
    def iou_continuous_loss(y_pred, y_true):
    	eps = 1e-6
    	def _sum(x):
    		return x.sum(-1).sum(-1)
    	numerator = (_sum(y_true * y_pred) + eps)
    	denominator = (_sum(y_true ** 2) + _sum(y_pred ** 2)
                  	  - _sum(y_true * y_pred) + eps)
    	return 1 - (numerator / denominator).mean()
    ```
    
11.  `nn.AdaptiveMaxPool2d(output_size)`   
    output_size 是pool后的大小，函数会自动计算池化的stride和kernel_size
    
12. `nn.Conv2d()`和`nn.functional.conv2d()` 中的 *dilation*, =2表示卷积点中间隔一个点。

13.  `nn.ConvTranspose2d()` 对于每一条边输出的尺寸：   
    output = (input-1)*stride + output_padding -2*padding + kernel_size   
    Unet中的使用：`self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)`
    
14.  `nn.Conv2d()` 带有dilation的输出尺寸为： `O = (H + 2*P - D*(F-1) -1)/S +1`   
    对于F=3的卷积核，Dilation与Padding一样才能保证SAME。
    
15.  `F.cross_entropy(input,target)`,其中input为NxCxHxW,而targe为NxHxW，*不需要onehot*. 需要**long()**类型   
    `CrossEntropyLoss = Log + Softmax() + NLLLoss`, **weights**的形状为[C] .    
    `BCELoss()`需要加上`sigmoid 或 softmax`，input为NxCxHxW,targe为NxCxHxW *需要onehot* , weight形状[nbatch]   
    `BCEWithLogitsLoss` 等于 `sigmoid() + BCELoss()`. **pos_weight**尺寸为[1]

16. `save(model)`要比`save(model.state_dict())`好很多

17. `with torch.no_grad():`用在inference中，节约内存

18. `grad_fn`表示该tensor是否是一个计算结果， `requires_grad` 表示是否要计算该tensor的梯度，默认false，**在网络训练中，输入的requires_grad 都是false，由于网络中间运算节点为True，因此最后的输出就自动求导了**。 `leaf variable`一般指事先创建的tensor

19. `tensor_A = tensor_B`， 把B赋值给A，A大小变了也会导致B边，它们指向同一块内存。可以用A=B+0，新建一块A

20. conv 和pooling在减小一般分辨率的时候一般设置为： `kernel=3` --> `padding=1`或 `kernel = 5` --> `padding=2`
21. `a.reshape(3,4)`可以不是contiguous的，相当于`a.contiguous()` + `a.view(3,4)`，`b.view(3,4)`则必须是contiguous的
22. `x.tranpose(0,1)`一次只能转换两个维度，`x.permute(1,2,0)`则可以用到高维上。它们的结果都是不contiguous的