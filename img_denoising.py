'''中值去噪python代码实现'''
import numpy as np
import os    
import skimage
from skimage import io

def load_image_skimage(filename,isFlatten=False):   #读取图像子函数       
    isExit=os.path.isfile(filename)    
    if isExit==False:        
        print("File open error!")    
    img=io.imread(filename)  #io.read(filename,img)读取文件    
    if isFlatten:        
        img_flatten=np.array(np.array(img,dtype=np.uint8).flatten())        
        return img_flatten   
    else:        
        img_arr=np.array(img,dtype=np.uint8)        
        return img_arr

def noising():
    image=load_image_skimage('image.jpg')
    image_gaussian=skimage.util.random_noise(image,'gaussian')  #添加高斯噪声
    image_salt=skimage.util.random_noise(image,'salt')          #添加盐噪声
    image_pepper=skimage.util.random_noise(image,'pepper')      #添加椒噪声
    skimage.io.imsave('image_gaussian.jpg',image_gaussian)
    skimage.io.imsave('image_salt.jpg',image_salt)
    skimage.io.imsave('image_pepper.jpg',image_pepper)

class denoising(object):     #定义领域平均化去噪类
    def __init__(self):
        self.filter=np.ones((3,3))     #定义卷积核
        self.zp=(self.filter.shape[0]-1)//2   #定义扩充大小
        self.size=self.filter.shape[0]*self.filter.shape[1]     #计算卷积核大小
        
    def padding(self,dic):    #填补子函数
        self.padding_dic=np.zeros((2*self.zp+dic.shape[0],2*self.zp+dic.shape[1],dic.shape[2]))
        self.padding_dic[self.zp:(dic.shape[0]+self.zp),self.zp:(dic.shape[1]+self.zp),:]=dic    
    
    def conv(self,filename):             #卷积子函数
        self.dic=load_image_skimage(filename,isFlatten=False)
        temp_dic=self.dic
        self.output=np.zeros((self.dic.shape[0],self.dic.shape[1],self.dic.shape[2]))  #创建输出矩阵
        for k in range(self.dic.shape[2]):
            for j in range(self.zp,self.dic.shape[1]+self.zp):
                for i in range(self.zp,self.dic.shape[0]+self.zp):
                    temp=temp_dic[(i-self.zp):(i+self.zp+1),(j-self.zp):(j+self.zp+1),k]
                    self.output[i-self.zp,j-self.zp,k]=np.median(temp)/255      #选区中值
        return self.output

if __name__=='__main__':
    func=denoising()
    noising()      #给图像分别加入高斯噪声、盐噪声、椒噪声
    output=func.conv('image_gaussian.jpg')
    skimage.io.imsave('imagenew_gaussian(3).jpg',output)   #高斯噪声图像处理
    output=func.conv('image_salt.jpg')
    skimage.io.imsave('imagenew_salt(3).jpg',output)       #盐噪声图像处理
    output=func.conv('image_pepper.jpg')
    skimage.io.imsave('imagenew_pepper(3).jpg',output)     #椒噪声图像处理