import numpy as np
import os    
from skimage import io

filename='image.jpg'

def load_image_skimage(filename,isFlatten=False):   #读取图像子函数   
    import os    
    from skimage import io    
    import numpy as np     
    isExit=os.path.isfile(filename)    
    if isExit==False:        
        print("File open error!")    
    img=io.imread(filename)  #io.save(filename,img)保存文件    
    if isFlatten:        
        img_flatten=np.array(np.array(img,dtype=np.uint8).flatten())        
        return img_flatten   
    else:        
        img_arr=np.array(img,dtype=np.uint8)        
        return img_arr

class Sharp(object):                #定义领域平均化去噪类
    def __init__(self):
        pass

    def Laplace_init(self):         #Laplace锐化初始化
        self.w1=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])    #定义卷积核
        self.zp=(self.w1.shape[0]-1)//2                 #定义扩充大小
        self.size=self.w1.shape[0]*self.w1.shape[1]     #计算卷积核大小
        self.k=1                                        #定义锐化系数K

    def Robert_init(self):          #Robert锐化初始化
        self.w1=np.array([[0,1],[-1,0]])    #定义卷积核
        self.w2=np.array([[1,0],[0,-1]]) 
        self.zp=(self.w1.shape[0]-1)//2                 #定义扩充大小
        self.size=self.w1.shape[0]*self.w1.shape[1]     #计算卷积核大小
        
    def padding(self,dic):          #填补子函数
        self.padding_dic=np.zeros((2*self.zp+dic.shape[0],2*self.zp+dic.shape[1],dic.shape[2]))
        self.padding_dic[self.zp:(dic.shape[0]+self.zp),self.zp:(dic.shape[1]+self.zp),:]=dic    
    
    def nomalization(self,dic):     #将图像标准化到[-1,1]区间内
        dic[:,:,:]=dic[:,:,:]*(dic[:,:,:]>0)
        dic[:,:,:]=dic[:,:,:] * (dic[:,:,:]<=255) + 255 * (dic[:,:,:]>255)
        dic[:,:,:]=dic[:,:,:]/255
        return dic
      
    def Laplace_conv(self):         #Laplace卷积子函数
        self.dic=load_image_skimage(filename,isFlatten=False)
        self.padding(self.dic)
        temp_dic=self.padding_dic   #将图像扩展为卷积尺度   
        temp_output=np.zeros((temp_dic.shape[0],temp_dic.shape[1],temp_dic.shape[2]))
        self.output=np.zeros((self.dic.shape[0],self.dic.shape[1],self.dic.shape[2]))  #创建输出矩阵
        for k in range(self.dic.shape[2]):
            for j in range(self.zp,self.dic.shape[1]+self.zp):
                for i in range(self.zp,self.dic.shape[0]+self.zp):
                    temp=temp_dic[(i-self.zp):(i+self.zp+1),(j-self.zp):(j+self.zp+1),k]
                    self.output[i-self.zp,j-self.zp,k]=np.sum(temp*self.w1)
        self.output=np.array((self.dic-self.k*self.output),dtype=np.uint8)
        #self.output=self.nomalization(self.output)
        io.imsave('LaplaceSharp.jpg',self.output)

    def Robert_conv(self):          #Robert卷积子函数
        self.dic=load_image_skimage(filename,isFlatten=False)
        self.padding(self.dic)
        temp_dic=self.padding_dic   #将图像扩展为卷积尺度   
        temp_output=np.zeros((temp_dic.shape[0],temp_dic.shape[1],temp_dic.shape[2]))
        self.output=np.zeros((self.dic.shape[0],self.dic.shape[1],self.dic.shape[2]))  #创建输出矩阵
        for k in range(self.dic.shape[2]):
            for j in range(self.zp,self.dic.shape[1]+self.zp):
                for i in range(self.zp,self.dic.shape[0]+self.zp):
                    temp=temp_dic[(i-self.zp):(i+self.zp+1),(j-self.zp):(j+self.zp+1),k]
                    self.output[i-self.zp,j-self.zp,k]=abs(np.sum(temp*self.w1))+abs(np.sum(temp*self.w2))
        self.output=self.nomalization(self.output)
        for k in range(self.output.shape[2]):
            for j in range(self.output.shape[1]):
                for i in range(self.output.shape[0]):
                    if(self.output[i,j,k]>0):
                        self.output[i,j,k]=1
                    else:
                        self.output[i,j,k]=-1
        io.imsave('RobertSharp.jpg',self.output)       


if __name__=='__main__':
    func=Sharp()
    #func.Robert_init()
    #func.Robert_conv()
    func.Laplace_init()
    func.Laplace_conv()