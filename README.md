# face_merging

环境配置 
opencv 3.3.0或以上版本
python 3.6.3
dlib 19.7
jupyter notebook

DelaunayTriangleCoodinate2Text文件
输入任何一张人脸图片和它对应的特征点坐标（本程序使用68个特征点坐标）。
在桌面生成tri.txt。
该文件是68个特征点构成的所有面部最小三角形的坐标集，用于三角面部融合。

face_merge文件
输入两张人脸图片，一张融合图一张模板图 ，alpha调节融合程度，数值越小第一张图融合成分越多。
输出两张图的融合图片和融合特征点。

face_swap文件
输入两张人脸图片，第一张模板图，第二张替代图。
输出替换面部后的图片。

face_merge_swap文件
是face_merge文件和face_swap文件的合并。最后生成融合后换脸的图片。

shape_predictor_68_face_landmarks.dat文件
用于检测人脸图片并生成68个特征点。
