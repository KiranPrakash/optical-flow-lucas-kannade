# Lucas Kannade Implementation for 3 Level Pyramid 

from numpy import linalg as LA
from scipy import signal
import numpy as num
import numpy as np 
from pylab import *
import matplotlib.pyplot as plt
import cv2

#1D Mask for the first Derivative of the Gaussian Kernel
def gaussDeriv(n,sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [-x / (sigma**3*sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

    
#Function returns the velocity vectors u and v by solving the matrices using Least Square Fit Method
def LucasKannadePyramid(Im1,Im2,corners,numlevels,iterations,window):
    
    #Generating Gaussian Pyramids for the Input image using cv2 inbuilt function
    for i in range(iterations):
        G= Im1.copy()
        gpA = [G]
        #processing all levels of the pyramid
        for i in xrange(numlevels):
            G= cv2.pyrDown(G)
            gpA.append(G)
    
    #returns image and time derivatives fx, fy, ft
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
    
    u = zeros((len(Im1[:,0]),len(Im1[0,:])))
    v = zeros((len(Im1[:,0]),len(Im1[0,:])))
    
    # Find the points only for the points that are good features to track(corners)
    for x in corners:
            j,i  = x.ravel()
            
            # this creates a 3x3 matrix for each pixel in the good feature point
            Fx = fx[(i-1):(i+2), (j-1):(j+2)]
            Fy = fy[(i-1):(i+2), (j-1):(j+2)]
            Ft = ft[(i-1):(i+2), (j-1):(j+2)]
           
            # transposed as i am using the ravel function which straightens up the 3x3 image to 9x1 reading columnwise   
            Fx= np.transpose(Fx)
            Fy= np.transpose(Fy)
            Ft= np.transpose(Ft)
                       
            Fx= Fx.ravel()
            Fy= Fy.ravel()
            Ft= -Ft.ravel()  
           
            #9x2 matrix
            A= np.transpose([Fx, Fy])
            
            #Least square fit to solve for unknown
            product1= np.dot(np.transpose(A),A)
            product2= np.dot(linalg.pinv(product1),np.transpose(A))
            U = np.dot(product2, Ft)
            
            #2x1 matrix containing
            u[i,j]=U[0]
            v[i,j]=U[1]
            
    return u,v
             
   
#Main Function to call that calls the LucasKannade function
def PlotLKPyr(Im1,Im2):
    
    #Using openCV inbuilt function goodFeaturestoTrack to find the corners in the image2
    corners = cv2.goodFeaturesToTrack(Im2,1000,0.01,10)
    corners = np.int0(corners)  
   
    
    u, v = LucasKannadePyramid(Im1,Im2,corners,3,9,1)
    
    #plotting the velocity vectors in the image2 
    plt.imshow(Im2, cmap = cm.gray)
    for i in range(len(u[:,0])):
        for j in range(len(u[0,:])):
            # thresholding the absolute value of the u and v
            if abs(u[i,j]) > 0.4 and abs(v[i,j]) > 0.15:
               
                ax = plt.axes()
                ax.arrow(j, i, u[i,j], v[i,j], head_width=4, head_length=5, color ='r')
    figure()
    plt.show()    


image1 = cv2.imread('basketball1.png',0)
image2 = cv2.imread('basketball2.png',0)
PlotLKPyr(image1, image2)

image3 = cv2.imread('teddy1.png',0)
image4 = cv2.imread('teddy2.png',0)
PlotLKPyr(image3, image4)

image5 = cv2.imread('grove1.png',0)
image6 = cv2.imread('grove2.png',0)
PlotLKPyr(image5, image6)                