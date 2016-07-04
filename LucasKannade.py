#Lucas-Kannade Optical Flow Estimation(without Pyramids, 1 level only)
#References: Implementation based on the OpenCV link provided in the webcourse and the 
#Optical Flow algorithm provided Shoaib Khan in CRCV UCF Website.

from numpy import linalg as LA
from scipy import signal
import numpy as num
import numpy as np 
from pylab import *
import matplotlib.pyplot as plt
import cv2

# We find the derivatives of the image in both the directions x and y and also derivative with respect to time
# We find the derivatives for both the gray scale image 1 and image 2



# Function returns image and time derivatives fx, fy, ft
def ImageGradients(Im1,Im2):
    #fx= ndimage.filters.gaussian_filter(Im1, 1.5, 1, mode= 'reflect') + ndimage.filters.gaussian_filter(Im2, 1.5, 1, mode= 'reflect')   
    
    # we perform the mask by predefining the gaussian derivative mask and convolve image1 and image2 in x
    x = [[-1,1], [-1,1]]
    gx=  0.25*np.array(x)
    fx = signal.convolve(Im1,gx) + signal.convolve(Im2,gx)
    fx.resize(len(Im1[:,0]),len(Im1[0,:]))
    
    # we perform the mask by predefining the gaussian derivative mask and convolve image1 and image2 in x       
    y = [[-1,-1], [1,1]]
    gy=  0.25*np.array(y)
    fy = signal.convolve(Im1,gy) + signal.convolve(Im2,gy)
    fy.resize(len(Im1[:,0]),len(Im1[0,:])) 

    # Time Derivative of image1 and image2
    gt = 0.25*ones((2,2))
    gt=  np.array(matrix(gt))  
    ft = signal.convolve(Im1,gt) + signal.convolve(Im2,-gt)
    ft.resize(len(Im1[:,0]),len(Im1[0,:]))
    
    return fx, fy, ft



        
#Function returns the velocity vectors u and v by solving the matrices using Least Square Fit Method
def LucasKannade(Im1,Im2,corners):
    
    fx, fy, ft = ImageGradients(Im1,Im2)
     
    u = zeros((len(Im1[:,0]),len(Im1[0,:])))
    v = zeros((len(Im1[:,0]),len(Im1[0,:])))
    
    # Solving for the velocity vectors only for the good Feature Points i.e corners detected using openCV function
    for x in corners:
            j,i  = x.ravel()
            
            #this creates a 3x3 matrix for each pixel in the good feature point
            Fx = fx[(i-1):(i+2), (j-1):(j+2)]
            Fy = fy[(i-1):(i+2), (j-1):(j+2)]
            Ft = ft[(i-1):(i+2), (j-1):(j+2)]
           
            #transposed as the ravel function would straighten up the 3x3 image to 9x1 reading columnwise   
            Fx= np.transpose(Fx)
            Fy= np.transpose(Fy)
            Ft= np.transpose(Ft)
                       
            Fx=  Fx.ravel()
            Fy=  Fy.ravel()
            Ft= -Ft.ravel()  
           
            #9x2 matrix
            A= np.transpose([Fx, Fy])
            
            #Least square fit to solve for unknown
            product1 = np.dot(np.transpose(A),A)
            product2 = np.dot(linalg.pinv(product1),np.transpose(A))
            U        = np.dot(product2, Ft)
            
            #2x1 matrix containing the velocity vector for a single pixel point
            u[i,j] = U[0]
            v[i,j] = U[1]
            
    return u,v
    
   
             
   
#Function to plot the optical FLow using Lucas Kannade
def PlotLK(Im1,Im2):
    
    #Using openCV inbuilt function goodFeaturestoTrack to find the corners in the image2
    corners = cv2.goodFeaturesToTrack(Im2,500,0.01,10)
    
    #detected corners converted to type int
    corners = np.int0(corners)
    
    
    u,v = LucasKannade(Im1, Im2, corners)
    
    #plotting the velocity vectors in the image2
    plt.imshow(Im2, cmap = cm.gray)
    for i in range(len(u[:,0])):
        for j in range(len(u[0,:])):
            # thresholding the absolute value of the u and v
            if abs(u[i,j]) > 0.5 and abs(v[i,j]) > 0.15:                                      
                ax = plt.axes()
                ax.arrow(j, i, u[i,j], v[i,j], head_width=4, head_length=5, color ='b')
    figure()
    plt.show()                    
                                           
                                
image1 = cv2.imread('basketball1.png',0)
image2 = cv2.imread('basketball2.png',0)
PlotLK(image1, image2)

image3 = cv2.imread('teddy1.png',0)
image4 = cv2.imread('teddy2.png',0)
PlotLK(image3, image4)

image5 = cv2.imread('grove1.png',0)
image6 = cv2.imread('grove2.png',0)
PlotLK(image5, image6)