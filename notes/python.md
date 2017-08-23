1. **skimg.io读取img**  读取的数组为：行*列*通道   依次为R,G,B 第4通道为亮度 
2. **win安装whl** 下载whl到/scripts ， cd到改目录 $pip install xxx.whl
3. **排序** np.sort(a, axis = 0) 按行排， 即排完每列由小到大  
           > np.sort(a, axis = 1) 按列排， 即排完每行由小到大