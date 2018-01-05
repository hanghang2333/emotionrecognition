需要完成的任务内容为:对图片里的人脸进行检测并对其表情进行分类。

项目代码地址:[代码地址链接](https://github.com/hanghang2333/emotionrecognition)

博客地址链接:[表情识别](http://www.jianshu.com/writer#/notebooks/14630615/notes/22134863)



####数据简介:

用的是FER-2013 这个数据库， 这个数据库一共有 35887 张人脸图像，来源于kaggle竞赛网站上下载的数据([数据下载地址](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data))。原始数据将标签和图片统一以csv格式存储，通过简单的转换后，可以得知图片大小均为48*48的灰度图(转换程序实例参见getpic.py)。下面是几个示例图片。该数据集标签为7类，分别为:

*0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'。

![0](/Users/lihang/code/fermodel/test/0.png)

![1](/Users/lihang/code/fermodel/test/1.png)

![2](/Users/lihang/code/fermodel/test/2.png)

![3](/Users/lihang/code/fermodel/test/3.png)

#### 数据预处理:

数据是竞赛数据，固定大小48*48，无需特别的预处理(除了归一化之类)。

####模型训练

网络结构用的是Xception网络结构,也是类似的使用了depthwise卷积的结构，(参见论文:Xception: Deep Learning with Depthwise Separable Convolutios).

具体的程序文件参看:cnn.py

优化方法使用Adam。

loss函数使用多分类softmax，后来尝试focal loss(针对样本难度不均衡问题使用)和其他一些变化，不过并没有带来提升。

初步测试集准确:0.66

### demo测试:

使用训练好的模型，完成测试。

要求:输入一张图片地址，输出这张图片里人脸框位置和对应人脸的表情(demo阶段为了看的方便将人脸框画在图片将表情结果也写在人脸上边)。

图片demo程序文件:image_demo.py.

测试结果:





![predicted_test_image](/Users/lihang/code/fermodel/images/predicted_test_image.png )

视频demo测试:

程序文件:video_demo.py.

这个就不再贴图了，读者自己运行下就大概知道了，其实就是实时获取摄像头图片而后分类并且显示在视频上。







