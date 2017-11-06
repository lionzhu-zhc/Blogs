1.	top表示输出， bottom表示输入
    2.Xavier 初始化方法
    3.if __name__ == 'main': { ...} 如果直接运行此.py则运行 {...}。 若其他文件调用则不运行{...}
    4.python import两种用法 尽量少用from xx import,建议 import xx as xxx， 防止模块名冲突
    5.with open(train_proto, 'w') as f:  with语句是提供一个有效的机制，让代码更简练，同时在异常产生时，清理工作更简单。
    6.sp={} {} 表示创建字典
    7.f.write('%s: %s\n' % (key, value))    %后接参数名
    8.caffe batch_normalization 层 后面要接scale层
    use_global_stats = 1; //如果为真，则使用保存的均值和方差，否则采用滑动平均计算新的均值和方差。
    // 该参数缺省的时候，如果是测试阶段则等价为真，如果是训练阶段则等价为假。
    moving_average_fraction = 2 [default = .999]; 
    // 滑动平均的衰减系数，默认为0.999
    eps = 3 [default = 1e-5];
    // 分母附加值，防止除以方差时出现除0操作，默认为1e-5
    9.conv中：
    num-output:卷积核的个数
    膨胀的卷积核尺寸 = 膨胀系数dilation * (原始卷积核尺寸 - 1) + 1
    group: 将对应的输入通道与输出通道数进行分组
    10.Eltwise层
    的操作有三个：product（点乘）， sum（相加减） 和 max（取大值）
