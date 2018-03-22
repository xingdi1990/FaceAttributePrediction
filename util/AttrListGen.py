import numpy as np
import os
import linecache as lc
from PIL import Image
import glob

filelist ='/home/labuser/PycharmProjects/attribute_classifier/datasets/celebA/mtcnn_align128/'
dicclebA = '/home/labuser/PycharmProjects/attribute_classifier/datasets/celebA/list_attr_celeba.txt'

arr = []
with open(dicclebA) as lmfile:
    lineNum=sum(1 for _ in lmfile)

image_list = []
for filename in sorted(glob.glob(filelist + '*.jpg')): #assuming gif
#    print(filename)
    image_list.append(filename)

#print(image_list)

print('total number of cropped images:', len(image_list))
it=iter(range(1, lineNum))
for m in it:
    line = lc.getline(dicclebA, m)
    Name = line.rstrip('\n')
    file = Name.split(" ")
    ImgName = '/home/labuser/PycharmProjects/attribute_classifier/datasets/celebA/mtcnn_align128/' + file[0]
    attr = Name[11:]
#    print(attr)
    arr.append(ImgName)
    arr.append(attr)

#print(arr)

for m in range(0, len(image_list)):
    if m % 10000 == 0:
        print('{} out of {}'.format(m, len(image_list)))
    ImgName = image_list[m]
    try:
        idx = arr.index(ImgName)
        attr = arr[idx+1]
        newline = ImgName + " " + attr
        with open('/home/labuser/PycharmProjects/attribute_classifier/datasets/celebA/detcelebAG',"a+") as outfile:
            outfile.write(newline + "\n")
    except:
        print(ImgName)
