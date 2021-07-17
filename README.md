# ChordioGram

### 一、项目介绍

本项目是对边缘特征描述符：弦直方图 相关论文的**非官方**复现。
此外对部分内容做了一些改进，使其能够适用于物体SLAM上。

相关论文包括：

+ Wang X, Zhang H, Peng G. A chordiogram image descriptor using local  edgels[J]. Journal of Visual Communication and Image Representation,  2017, 49: 129-140.
+ Toshev A, Taskar B, Daniilidis K. Shape-based object detection via  boundary structure segmentation[J]. International journal of computer  vision, 2012, 99(2): 123-146.
+ 王小龙. 基于特征组合的视觉闭环检测研究[D]. 西北工业大学, 2018.

### 二、使用及测试

```bash
mkdir build && cd build
cmake ..
make
../bin/main ../data/1.jpg ../data/2.jpg ../data/1.txt ../data/2.txt
../bin/main ../data/3.jpg ../data/4.jpg ../data/3.txt ../data/4.txt
../bin/main ../data/4.jpg ../data/5.jpg ../data/4.txt ../data/5.txt
../bin/main ../data/10.png ../data/11.png ../data/10.txt ../data/11.txt
../bin/main ../data/11.png ../data/12.png ../data/11.txt ../data/12.txt
```

