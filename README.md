# Chinese-Plate-recognition_tensorflow

## 本项目主要实现端到端车牌识别
http://blog.csdn.net/ssmixi/article/details/78220039

http://blog.csdn.net/ssmixi/article/details/78223907

训练样本全来自仿真数据，训练24万张样本。验证集上识别精度为82%。以下为测试32张的结果为0.93。

![image](https://github.com/sheirving/Chinese-Plate-recognition_tensorflow/blob/master/result/32%E5%BC%A0%E5%9B%BE%E6%B5%8B%E8%AF%95%E7%BB%93%E6%9E%9C.png)

错误类型主要为噪音太强、模糊、半遮挡以及相似字符（具体可见博客）。

## 运行方式：

1.Noplates,font,images均为genplate.py运行所需文件。

2.训练模型：
  cnn_model.py(神经网络模型)
  input_data.py(输出训练数据，此处并未将仿真数据生成为图片保存到硬盘，后在读取给网络，而是直接生成batch送到网络）
  运行： train.py(生成训练模型）
  
3.测试模型：
  gen_txt.py(主要是将genplate.py生成的测试样本，转换成一个txt文本，记录文本路径及标签）
  eval_one_image.py(测试一张图）
  eval_batch_samples.py(批量测试）
  test.py(不用理，仅为测试代码)
  

