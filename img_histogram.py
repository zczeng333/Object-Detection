"""直方图均衡化python代码实现"""
from PIL import Image
import pylab as pl
import numpy as np


def histeq(im,nbr_bins = 256):  #直方图均衡化子函数
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   #获取原图像的灰度值概率分布(完成概率密度积分)
    cdf = 255.0 * cdf / cdf[-1]     #将[0,1]的概率分布函数转换为[0,255]范围内
    im2 = np.interp(im.flatten(),bins[:-1],cdf)    #对于原图像像素im,依据函数cdf=f(bins)进行线性插值(映射)
    return im2.reshape(im.shape),cdf


pil_im = Image.open('image.bmp')        #打开原图
pil_im_gray = pil_im.convert('L')       #转化为灰度图像
pil_im_gray.show()         #显示灰度图像

im = np.array(Image.open('image.bmp').convert('L'))
pl.figure()
pl.hist(im.flatten(),256,color='black')  #显示原图像灰度直方图
pl.xlabel('pixel')
pl.ylabel('possibility')

im2,cdf = histeq(im)
pl.figure()
pl.hist(im2.flatten(),256,color='black') #显示直方图均衡后图像灰度直方图
pl.xlabel('pixel')
pl.ylabel('possibility')
pl.show()

im2 = Image.fromarray(np.uint8(im2))
im2.show()
im2.save("imagenew.bmp")