import numpy as np 
import math

####Loss function part
yhat = np.array([.3, 0.2, 0.1, .8, .7])
y = np.array([1, 0, 0, 1, 1])
print("Loss Function 1",sum(abs(yhat-y)))
yhat = np.array([0, 0, 0.1, 0.6, 0.2])
y = np.array([1, 0, 0, 1, 1])
print("Loss Function 1",sum(abs(yhat-y)))


yhat = np.array([.3, 0.2, 0.1, .8, .7])
y = np.array([1, 0, 0, 1, 1])
x=yhat-y
x = np.dot(x,x)
print("Loss Function 2",x)


yhat = np.array([0, 0, 0.1, 0.6, 0.2])
y = np.array([1, 0, 0, 1, 1])
x=yhat-y
x = np.dot(x,x)
print("Loss Function 2",x)

###Sigmoid part

import matplotlib.pyplot as plt 
import numpy as np 
import math 


def sigmoid(number):
    return 1/(1 + np.exp(-number))
def dsigmoid(number):
    f = 1/(1 + np.exp(-number))
    return f * (1 - f)

number=10
vector =  np.array([12,4,2,0])
matrix=  np.array([[5, 78, 2, 34, 0],
 [6, 79, 3, 35, 1],
 [7, 80, 4, 36, 2]])


print("Number",number)
print("Sigmoid",sigmoid(number))
print("Derivative",dsigmoid(number))

print("Vector",vector)
print("Sigmoid",sigmoid(vector))
print("Derivative",dsigmoid(vector))

print("Matrix",matrix)
print("Sigmoid",sigmoid(matrix))
print("Derivative",dsigmoid(matrix))


#Image part
import skimage.io
import skimage.viewer

def viewimage(image):
    viewer = skimage.viewer.ImageViewer(image)
    viewer.show()
    
##This function will take path and filename and it will return the array representation of image.
    
def readimage(path,filename):
    image = skimage.io.imread(fname=path+"/"+filename)
    return image

def getproperties(image):
    print("Vectorized shape of Image", image.shape)
    print("Number of channels",image.ndim)
    print("Number of column",image.shape[1])
    print("Number of rows",image.shape[0])

    
    
seven =readimage('C:/Users/Muhammad Muneeb/Desktop/MSSEMESTER3/Deeplearning/Assignments','img_1.jpg')
cat =readimage('C:/Users/Muhammad Muneeb/Desktop/MSSEMESTER3/Deeplearning/Assignments','img_2.jpg')

getproperties(seven)
getproperties(cat)



#########Updated Version
print(seven.shape)
seven = seven.reshape(-1,1)
print(seven.shape)

print(cat.shape)
cat = cat.reshape(-1,1)
print(cat.shape)




#from skimage.color import rgb2gray
#grayscale = rgb2gray(cat)
#getproperties(grayscale)


#viewer = skimage.viewer.ImageViewer(seven)
#viewer.show()
