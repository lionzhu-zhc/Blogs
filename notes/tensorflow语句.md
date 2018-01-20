1. Tensorboard : cd到logs文件夹的上一个文件夹，然后terminal输入 $ tensorboard --logdir='logs/', 

   在chrome中转入localhost:6006

2. `tensor .get_shape()` 得到形状;    `type(AA)` 得到数据类型

3. `conv2d_tranpose(,filters, ...)` filter 也就是卷积核的weights的shape 为[height, width,**out_channels, in_channels**] ,与conv2d filter 的参数顺序[height, width, out_channels,  in_channels]中inchannels和outchannels顺序相反，并且`conv2d_tranpose`还多一个参数：output_shape

4. sess.run([a, b]) 就是直接返回run的结果，类型是ndarr，用于查看tensor的值

5. tf保存模型：  saver.save(sees, 'path', write_meta_graph = False )

   ​			tf.reset_default_graph()    # destroy pervious net

   ​			saver.restore(sees, 'path')

6. tfrecords 读出来的batch 需要先np.transpose([0, 2, 1]), 再reshape( 128, 128, 64)

7. ```Tf.nn.softmax_cross_entropy_with_logits()```, *logits* 是神经网络最后一层特征输出  
      它先对logits做softmax，再对softmax结果做cross_entropy   

      `Tf.nn.sparse_softmax_cross_entropy_with_logits()` 比`softmax_cross_entropy_with_logits()`多了一步将*labels* 稀疏化的操作

8. 同一个session里只能用一次 `tf.train.start_queue_runners()` ，且不可以加coordinator

9. `func(2, y = 3)` 前面的关键字可以省略，不能前面参数有关键字而后者没有关键字。变量在都有关键字时可以换顺序

10. ​`tf.summary.scalar('name', var)`   
	`tf.summary.histogram('h1/weights',w_2)`
   `merge_op = tf.summary.merge_all()`   
   `writer = tf.summary.Filewriter('/dir', sess.graph)`   
   `writer.add_summary(summary_str,itr)` 把迭代步骤绑定到graph中