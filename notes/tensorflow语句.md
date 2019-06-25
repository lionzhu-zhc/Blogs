1. Tensorboard : cd到logs文件夹的上一个文件夹，然后terminal输入 $ tensorboard --logdir='logs/', 

   在chrome中转入localhost:6006

2. `tensor .get_shape()` 得到形状;    `type(AA)` 得到数据类型

3. `conv2d_tranpose(,filters, ...)` filter 也就是卷积核的weights的shape 为[height, width,**out_channels, in_channels**] ,与conv2d filter 的参数顺序[height, width, in_channels,  out_channels]中inchannels和outchannels顺序相反，并且`conv2d_tranpose`还多一个参数：output_shape

4. sess.run([a, b]) 就是直接返回run的结果，类型是ndarr，用于查看tensor的值

5. tf保存模型：
   ``` python
    saver.save(sees, ..., write_meta_graph = False )

    tf.reset_default_graph()    # destroy pervious net

    saver.restore(sees, 'path') 
   ```
6. tfrecords 读出来的batch 需要先np.transpose([0, 2, 1]), 再reshape( 128, 128, 64)

7. ```Tf.nn.softmax_cross_entropy_with_logits()```, *logits* 是神经网络最后一层特征输出  
      它先对logits做softmax，再对softmax结果做cross_entropy   

      `Tf.nn.sparse_softmax_cross_entropy_with_logits()` 比`softmax_cross_entropy_with_logits()`多了一步将*labels* 稀疏化的操作
      `tf.losses.sparse_softmax_cross_entropy` logits的shape为[batch,depth,width,height,channel]， 而label的shape为[batch,depth,width,height]，**比logits少了channel维度**

8. 同一个session里只能用一次 `tf.train.start_queue_runners()` ，且不可以加coordinator

9. `func(2, y = 3)` 前面的关键字可以省略，不能前面参数有关键字而后者没有关键字。变量在都有关键字时可以换顺序

10. ​`tf.summary.scalar('name', var)`   
   `tf.summary.histogram('h1/weights',w_2)`
   `merge_op = tf.summary.merge_all()`   
   `writer = tf.summary.Filewriter('/dir', sess.graph)`   
   `writer.add_summary(summary_str,itr)` 把迭代步骤绑定到graph中
   > https://zhuanlan.zhihu.com/p/26203726  

11.  numpy 读取的raw行列相反，语句为：由此得到 [512, 512, 8]
```python
A = np.fromfile(path, dtype='float64', sep='')
B = A.reshape([8,512,512])
C = np.transpose(B, (2,1,0))
```
12.  a的shape为 *2x2x3*
```python
a = tf.constant([[[0.8, 0.1, 0.1], [0.2, 0.2, 0.3]],[[0.4, 0.4, 0.5], [0.6, 0.6, 0.7]]])
```
[0.8, 0.1, 0.1]表示的是坐标为(0,0,n)的点，  
[0.2, 0.2, 0.3]表示的是坐标为(0,1,n)的点。。。

a 的第一层：  

| 0.8  | 0.2  |
| ---- | ---- |
| 0.4  | 0.6  |

a 的第二层：

| 0.1  | 0.2  |
| ---- | ---- |
| 0.4  | 0.6  |


a 的第三层：

| 0.1  | 0.3  |
| ---- | ---- |
| 0.5  | 0.7  |


`b= tf.argmax(a, axis = 2)` b的结果shape为*2x2*，结果为：

| 0    | 2    |
| ---- | ---- |
| 2    | 2    |

表示每个(x,y)坐标对应的channel值最大的那个channel的index，即该点分类的目标类标

13.  `tf.shape(tensor)` 返回到结果是[行，列，深]，如果是1维的向量，则只返回[列]的值  
    `tf.reshape(tensor, (-1,))` 将数组扁平化，变为*1xN*的向量

14.  TF restore ckpt, 不重复定义网络结构的话，调用
```python
    saver = tf.train.import_meta_graph("save/model.ckpt.meta")
    with tf.Session() as sess:
        saver.restore(sess, "save/model.ckpt")
```
ckpt的目录中，*meta*是模型图结构，*index*是参数索引，*data*是数据。一般model-1000.ckpt会自动加上迭代次数

15.  tf graph 的用法  
```python
g1 = tf.Graph()  
with g1.as_default():  
    c1 = tf.constant(4.0)  
  
g2 = tf.Graph()  
with g2.as_default():  
    c2 = tf.constant(20.0)  
  
with tf.Session(graph=g1) as sess1:  
    print(sess1.run(c1))  
with tf.Session(graph=g2) as sess2:  
    print(sess2.run(c2)) 
```
结果为 4.0， 20.0


16. 打印tf的变量名和变量值 tf.all_variables() , tensor 中有属性.name， run一下得到values
```python
variable_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variable_names)
for k,v in zip(variable_names, values):
    print("Variable: ", k)
    print("Shape: ", v.shape)
    print(v)
```
17. tf fine-tuning
```python
saver = tf.train.import_meta_graph('vgg.meta')
# 访问图
graph = tf.get_default_graph() 

#访问用于fine-tuning的output
fc7= graph.get_tensor_by_name('fc7:0')

#如果你想修改最后一层梯度，需要如下
fc7 = tf.stop_gradient(fc7) # It's an identity function
fc7_shape= fc7.get_shape().as_list()

new_outputs=2
weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
output = tf.matmul(fc7, weights) + biases
pred = tf.nn.softmax(output)

# Now, you run this with fine-tuning data in sess.run()
```
> https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-16-transfer-learning/
> https://zhuanlan.zhihu.com/p/44904619
***
18. `tf.Variable()` 每次都会创建新的变量，无法共享变量   
    `tf.get_variable()` 结合作用域可知若指定的变量名已存在则返回存在的变量，否则新建变量

19. `tensor.get_shape().as_list()[-1]` 取tensor的最后一维大小，可直接得到，不需要`sess.run()`, `get_shape()`得到的是tensor的tuple类型,是静态维度，用于placehoder都是固定的情况  
    `b= tf.shape(a)` 返回的是一个tensor，需要sess.run(b)得到ndarray, 其中a可以是*tensor, list, array*，得到动态维度，可用于[none, none, 1]这种情况

20. `tf.nn.softmax(Tensor)` 返回跟Tensor一样shape的tensor，其值为根据原Tensor各channel值计算得到的概率值

21. 旧版查看ckpt的模型的变量：   
```
from tensorflow.python.tools.inspect_checkpoint import  print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file('.../model.ckpt', tensor_name= None, all_tensors= False, all_tensor_names= True)
文件目录有model.ckpt.index等
```
新版ckpt   
```
print_tensors_in_checkpoint_file('.../name', tensor_name= None, all_tensors= False, all_tensor_names= True)
文件只有name.index等
```

22. 根据名字加载ckpt变量
```
#得到该网络中，所有可以加载的参数   
variables = tf.contrib.framework.get_variables_to_restore()   
#删除output层中的参数   
variables_to_resotre = [v for v in varialbes if v.name.split('/')[0]!='output']   
#构建这部分参数的saver   
saver = tf.train.Saver(variables_to_restore)saver.restore(sess,'model.ckpt')
```
>https://blog.csdn.net/b876144622/article/details/79962727?utm_source=copy

23. tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量   

tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量
>https://blog.csdn.net/UESTC_C2_403/article/details/72328815 

