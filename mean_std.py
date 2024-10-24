
import numpy as np
import cv2
import os
 
# img_h, img_w = 32, 32
img_h, img_w = 256, 256  #根据自己数据集适当调整，影响不大
meanAs, stdevAs, meanBs, stdevBs  = [], [], [], []
imgA_list = []
imgB_list = []
 
imgs_path = '/temp8/contest/hrx/data/crop'
imgsA_path_list = []
imgsB_path_list = []
f = open("/temp8/contest/hrx/data/crop/list/train.txt")
content = f.read()


lines = content.split('\n')
file_list = []

for line in lines:
    file_list.append(line)
file_list.pop()
print(len(file_list))

for dirpath,dirnames,filenames in os.walk(imgs_path):
    # print(dirpath)
    if 'label' in dirpath:
        continue
    #if 'train' in dirpath:
    if 'before' in dirpath:
        for filename in filenames:
            # print(filename)
            if filename[7:] in file_list:
                print(filename)
            #if 'png' in filename:
                imgsA_path_list.append(os.path.join(dirpath,filename))
    elif 'after' in dirpath:
        for filename in filenames:
            if filename[6:] in file_list:
                print(filename)
        # for filename in filenames:
        #     if 'png' in filename:
                imgsB_path_list.append(os.path.join(dirpath,filename))
# imgs_path_list = os.listdir(imgs_path)

len_ = len(imgsA_path_list) + len(imgsB_path_list)
# print(imgs_path_list)
i = 0
for item in imgsA_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    imgA_list.append(img)
    i += 1
    print(i,'/',len_)    
 
for item in imgsB_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    imgB_list.append(img)
    i += 1
    print(i,'/',len_)  
imgAs = np.concatenate(imgA_list, axis=3)
imgBs = np.concatenate(imgB_list, axis=3)
imgAs = imgAs.astype(np.float32) / 255.
imgBs = imgBs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgAs[:, :, i, :].ravel()  # 拉成一行
    meanAs.append(np.mean(pixels))
    stdevAs.append(np.std(pixels))

    pixels = imgBs[:, :, i, :].ravel()  # 拉成一行
    meanBs.append(np.mean(pixels))
    stdevBs.append(np.std(pixels))
 
# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
meanAs.reverse()
stdevAs.reverse()
meanBs.reverse()
stdevBs.reverse()
 
print("normMeanA = {}".format(meanAs))
print("normStdA = {}".format(stdevAs))

print("normMeanB = {}".format(meanBs))
print("normStdB = {}".format(stdevBs))
