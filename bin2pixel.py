import numpy
import pandas as pd
from PIL import Image
import os
import sys
import binascii
from tqdm import tqdm
import cv2
width_dict = {10:32,30:64,60:128,100:256,200:384,500:512,1000:768}
def getMatrix_From_Bin():
    width = 1024

    #修改成apk文件路径
    path = 'test/'


    tempfiles = os.listdir(path)
    for file in tqdm(tempfiles):
        width = 1024
        for widd in width_dict:
            if int(os.path.getsize(os.path.join(path,file))/1024) > widd:
                pass
            else:
                width = width_dict[widd]
        file_content = open(os.path.join(path,file),'rb')
        content=file_content.read()
        hexst=binascii.hexlify(content)
        fh=numpy.array([int(hexst[i:i+2],16)for i in range(0,len(hexst),2)])
        rn=len(fh)/width
        fh=numpy.reshape(fh[:int(rn)*width],(-1,width))
        fh=numpy.uint8(fh)
        im = Image.fromarray(fh)

        #写图片保存路径
        im.save(os.path.join(os.getcwd(),file.split('.')[0]+'.png'))



if __name__=='__main__':
    # getMatrix_From_Bin()
    # path = 'test2/'
    # tempfiles = os.listdir(path)
    # for file in tempfiles:
    #     print(file)
    #     filename = path + file
    #     image = cv2.imread(filename)
    #     print(image)
    width = 1024

    #修改成apk文件路径
    path = './feature_vectors/'


    tempfiles = os.listdir(path)
    count = 0
    for file in tqdm(tempfiles):
        file_content = open(os.path.join(path,file),'rb')
        
        content = file_content.read()
        hexst = binascii.hexlify(content)
        fh = numpy.array([int(hexst[i:i+2],16)for i in range(0,len(hexst),2)])

        if count < fh.shape[0]:
            count = fh.shape[0]
    print(count)

