1. pip freeze: 查看所安装的包
2. sudo pip install - -upgrade numpy 更新
3. pip - -version 查看版本
4. caffe 画loss曲线     
	将 caffe-master/tools/extra/parse_log.sh  caffe-master/tools/extra/extract_seconds.py和 caffe-master/tools/extra/plot_training_log.py.example 拷到log目录
        在训练.sh中加入保存log的语句   
	转到log目录， 执行$./parse_log.sh caffe.wujiyang-ubuntu.wujiyang.log     ./plot_training_log.py.example 6 demo.png caffe.wujiyang-ubuntu.wujiyang.log 
    0: Test accuracy  vs. Iters  
    1: Test accuracy  vs. Seconds  
    2: Test loss  vs. Iters  
    3: Test loss  vs. Seconds  
    4: Train learning rate  vs. Iters  
    5: Train learning rate  vs. Seconds  
    6: Train loss  vs. Iters  
    7: Train loss  vs. Seconds    
5. $nvidia-smi 
6. $top 查看进程
7. scp santiago@223.3.54.242:/raid/santiago/lits/Mat/KidneyTrainMat/657737_1_high_L-99-4.mat ./
8. ssh -2 santiago@223.3.54.242 