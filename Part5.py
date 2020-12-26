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



#from skimage.color import rgb2gray
#grayscale = rgb2gray(cat)
#getproperties(grayscale)


viewer = skimage.viewer.ImageViewer(seven)
viewer.show()