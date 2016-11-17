import cv2
import numpy as np
import imutils
import math
        
i = 0

class image :
        
        def __init__(self , src):


                                                global i
                                                self.img = cv2.imread(src, 0)
                                                self.img = imutils.resize(self.img, 1920 ,1080)

                                                self._ , self.test = cv2.threshold(self.img ,177 , 255 , cv2.THRESH_BINARY_INV)

                                                #collecting the indices of ON pixels along the y axis(top down) as an ndarray
                                                self.max_values_y = np.argmax(self.test , axis = 0)


                                                #changing all OFF pixels to max to remove interference while
                                                #searching for first ON pixel along y axis
                                                self.max_values_y[self.max_values_y == 0] = self.test.shape[0]
                                                self.roi_y_min = self.max_values_y.min()

                                                #Searching for the last ON pixel along y axis
                                                self.max_values_y_reverse = np.argmax(self.test[::-1,:] , axis = 0)
                                                self.max_values_y_reverse[self.max_values_y_reverse == 0] = self.test.shape[0]
                                                self.roi_y_max = self.test.shape[0] - self.max_values_y_reverse.min()

                                                #Collecting indices of first ON pixels along the x axis left to right
                                                self.max_values_x = np.argmax(self.test , axis = 1)
                                                self.max_values_x[self.max_values_x==0] = self.test.shape[1]
                                                self.roi_x_min = self.max_values_x.min()

                                                #Colecting indices of ON pixels along x axis right to left
                                                self.max_values_x_reverse = np.argmax(self.test[:,::-1] , axis = 1)
                                                self.max_values_x_reverse[self.max_values_x_reverse == 0] = self.test.shape[1]
                                                self.roi_x_max = self.test.shape[1] - self.max_values_x_reverse.min()


                                                self.final_image = self.test[self.roi_y_min:self.roi_y_max, self.roi_x_min:self.roi_x_max]


                                
                                                #Finding the aspect ratio given by width / height
                                                #.shape function gives the y and x coordinates of the bounding box of the cropped image
                                                self.aspectratio = float(self.final_image.shape[1])/float(self.final_image.shape[0])
                                                                                
                                                
                                                #Calculating the Normalised Area
                                                self.temp_image = self.final_image.copy()
                                                self.temp_image[self.temp_image == 255] = 1    #Sets all the ON pixels to 1 to calculate the area 
                                                self.pixel_area = self.temp_image.sum()
                                                self.normalised_area = float(self.pixel_area)/(self.final_image.size)
                           
                                
                                                #calculating the X and Y coordinates of the COG of the image
                                                self.Y_coordinates , self.X_coordinates = self.temp_image.nonzero()
                                                self.X_COG = self.X_coordinates.sum()/self.X_coordinates.size
                                                self.Y_COG = self.Y_coordinates.sum()/self.Y_coordinates.size
                                

                                                #Calculating the normalised signature height
                                                self.heights_begining = np.argmax(self.final_image , axis = 0)

                                                self.heights_ending  = np.argmax(self.final_image[::-1,:] ,axis =0)
                                                self.heights_ending = self.final_image.shape[0] - self.heights_ending

                                                self.heights = self.heights_ending - self.heights_begining + 1
                                                self.H_max = self.heights.max()


                                                self.width_beginning = np.argmax(self.final_image , axis = 1)
                                                self.width_ending = np.argmax(self.final_image[: , ::-1] , axis = 1)
                                                self.width_ending = self.final_image.shape[1] - self.width_ending

                                                self.widths = self.width_ending - self.width_beginning + 1
                                                self.W_max = self.widths.max()
                                                self.normalised_signature_height = float(self.H_max)/float(self.W_max)
                                                                                
                                                #Calculating the baseline Shift
                                                #Difference in the Y_COG of the left and right half of the image divided 
                                                #based on the COG
                                                
                                                #Splitting the image into left and right halves
                                                self.left_ = self.final_image[:,0:self.X_COG]
                                                self.right_ = self.final_image[:,self.X_COG:]


                                                self.left_Y_COG , self._ = self.left_.nonzero()

                                                self.left_Y_COG = self.left_Y_COG.sum()/self.left_Y_COG.size

                                                self.right_Y_COG , self._ = self.right_.nonzero()
                                                self.right_Y_COG = self.right_Y_COG.sum()/self.right_Y_COG.size

                                                self.baseline_shift = abs(self.right_Y_COG-self.left_Y_COG)

                                                print self.final_image.shape

                                                cv2.imshow("final"+str(i) , self.final_image)
                                                i+=1
                
        def display_data(self):
                print "Aspect Ratio",self.aspectratio
                print "Normalised Area" , self.normalised_area
                print "Centre of Gravity(Y,X)", self.Y_COG , self.X_COG
                print "Normalised Signature height", self.normalised_signature_height
                print "Baseline Shift" , self.baseline_shift
                print "Maximum Height" , self.H_max 
                print "Maximum width" , self.W_max
                
                    

class feature_vector:
            
                def __init__(self, image):
                    self.vector = np.array([image.aspectratio , image.normalised_area ,image.X_COG ,image.Y_COG ,image.normalised_signature_height, image.baseline_shift, image.H_max, image.W_max])
                            

def Euclidean_Distance(vector ,  mean_signature):
                i = 0
                Euclidean_Distances = list()
                while i < len(vector):
                       Euclidean_Distances.append(math.sqrt(((vector[i].vector-mean_signature)**2).sum()))
                       i+=1
                return Euclidean_Distances              
                                                                                 



image_sources = list()
vector = list()
img = list()
Mean_Signature = np.zeros(8)

#Creating a list of all the image file names 
for j in range(19):
     image_sources.append("sign"+str(j)+".jpg")


#Creating a list of image objects based on the sources 
for j in image_sources:
        img.append(image(j))


#Creating a list of feature vectors 
for j in img:
        vector.append(feature_vector(j))



#Calculating the mean signature 
for j in vector:
        Mean_Signature += j.vector
        
Mean_Signature /= len(vector)


x = Euclidean_Distance(vector , Mean_Signature)
Tolerance = max(x)

#######################################################################################
###############################Testing Starts##########################################


try :
        test_image = image(raw_input("Enter the file name of the test file ")+".jpg")
        test_features = feature_vector(test_image)
        test_Euclidean_distance = math.sqrt(((test_features.vector - Mean_Signature)**2).sum())
        print "Tolerance", Tolerance
        print "Test_Distance" , test_Euclidean_distance

        if test_Euclidean_distance <= Tolerance :
                print "Verified"
        else:
                print "Forged"
   
except :
        print("File not found. Terminating")

finally:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
	

   
		
                

                

