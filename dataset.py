#coding=utf8
import codecs
import cv2
import numpy as np
from maketwo import Detecttwo
from tqdm import tqdm
datapath = '/unsullied/sharefs/lh/isilon-home/fer2013/ferpic/'
label = '/unsullied/sharefs/lh/isilon-home/fer2013/label.txt'
label = codecs.open(label,'r','utf8').readlines()
label = [i.split('\t') for i in label]
label = [  [i[0] ,int(i[1]) ] for i in label]
#print(label[0:5])
X = []
Y = []
for i in tqdm(label):
    picname = datapath+i[0]+'.png'
    img = cv2.imread(picname,0)
    img = np.expand_dims(img,axis=2)#224*224*1
    img = cv2.resize(img, (48,48), interpolation=cv2.INTER_LINEAR)
    #cv2.imwrite('1.png',img)
    img = np.expand_dims(img,axis=2)#224*224*1
    X.append(img)
    y = [0,0,0,0,0,0,0]
    y[i[1]]=1
    y = np.array(y)
    Y.append(y)
X = np.array(X)
Y = np.array(Y)
print(X.shape,Y.shape)
print(X[0],Y[0:5])

import h5py
f = h5py.File("Data.hdf5",'w')
f.create_dataset('X',data=X)
f.create_dataset('Y',data=Y)
f.close()
#np.save('X.npy',X)
#np.save('Y.npy',Y)