#coding=utf8
import codecs
import cv2
from tqdm import tqdm
import numpy as np
f = codecs.open('fer2013.csv','r','utf8').readlines()[1:]
labelfile = codecs.open('label.txt','w','utf8')
index = 0
for line in tqdm(f):
    flist = line.split(',')
    label = flist[0]
    img = flist[1].split(' ')
    img = [int(i) for i in img]
    img = np.array(img)
    img = img.reshape((48,48))
    cv2.imwrite('ferpic/'+str(index)+'.png',img)
    labelfile.write(str(index)+'\t'+label+'\n')
    index += 1