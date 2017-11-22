1. Tensorboard : cd到logs文件夹的上一个文件夹，然后terminal输入 $ tensorboard --logdir='logs/', 

   在chrome中转入localhost:6006

2. tensor .get_shape() 得到形状;    type(AA) 得到数据类型

3. conv3d_tranpose(,filters, ...) filter 的shape与conv3d filter shape的inchannels和outchannels顺序相反

4. sess.run([a, b]) 就是直接返回run的结果，用于查看tensor的值

5. tf保存模型：  saver.save(sees, 'path', write_meta_graph = False )  

   ​			tf.reset_default_graph()    # destroy pervious net

   ​			saver.restore(sees, 'path')  
6. tfrecords 读出来的batch 需要先np.transpose([0, 2, 1]), 再reshape( 128, 128, 64)