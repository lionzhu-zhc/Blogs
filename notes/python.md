1. **skimg.io读取img**  读取的数组为：行*列*通道   依次为R,G,B 第4通道为亮度 
2. **win安装whl** 下载whl到/scripts ， cd到改目录 $pip install xxx.whl
3. **排序** np.sort(a, axis = 0) 按行排， 即排完每列由小到大  
           &emsp;&emsp; np.sort(a, axis = 1) 按列排， 即排完每行由小到大

4. python 文件结构  
   ![1](https://github.com/lionzhu6336/Blogs/raw/master/notes/python_1.PNG)  
```
def main():  
if __name__ == "__main__":  
 此处即是main的入口
```
5. python  array保存png图像

   ​scipy.misc.toimage(array, cmin=0, cmax =255).save('path')
   数组顺序为 RGB

6. Spicy.io.loadmat 读数组的顺序是[column, row, slice]
7. `p.full((3,4), 10)` 创建3x4矩阵，初始值为10
8. `cord = np.where(data==0)` 查找data中为0的位置，`data[cord[0], cord[1]] = 1` 替换成1