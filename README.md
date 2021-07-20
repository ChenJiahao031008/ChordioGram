# ChordioGram

### 一、项目介绍

本项目是对边缘特征描述符：弦直方图 相关论文的**非官方**复现。
此外对部分内容做了一些改进，并加入了YOLOv5-LibTorch版本实时检测物体，使其能够在未来应用于物体SLAM上。

相关论文包括：

+ Wang X, Zhang H, Peng G. A chordiogram image descriptor using local  edgels[J]. Journal of Visual Communication and Image Representation,  2017, 49: 129-140.
+ Toshev A, Taskar B, Daniilidis K. Shape-based object detection via  boundary structure segmentation[J]. International journal of computer  vision, 2012, 99(2): 123-146.
+ 王小龙. 基于特征组合的视觉闭环检测研究[D]. 西北工业大学, 2018.

### 二、使用及测试

**配置说明**：

由于涉及到深度学习，这块比较麻烦一点。我的配置如下：

+ RTX 3060 Laptop显卡 Driver 470.42.01
+ CUDA 11.1（显卡太新以至于11以下的不兼容）
+ cudnn 8.1.0.77
+ OpenCV 3.4.4 （无cuda版本，因为不支持cuda11）
+ LIbTorch： libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111 

运行前修改`CMakeLists.txt`文件中libtorch库的位置（第13行）

**编译运行测试**

```bash
mkdir build && cd build
cmake ..
make
../bin/main ../data/01.jpg ../data/02.jpg
../bin/main ../data/04.jpg ../data/07.jpg  
```

### 三、参考

YOLOv5的libtorch版本参考仓库：https://github.com/Nebula4869/YOLOv5-LibTorch
