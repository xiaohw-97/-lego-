# -lego积木探测-
运用各类传统计算机视觉算法探测不同种类乐高积木数量，可利用实力图片进行测试
使用案例：
rgbImage = imread('train04.jpg');
[redone, blueone] = count_lego(rgbImage);
