#importing some useful packages
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import cv2
import os
#import cv2
#from PIL import Image
#import math
import fnmatch
import glob
import pickle
'''
Part 1 - Camera Calibration and Calculating Distortion Correction
'''
images = glob.glob(os.getcwd() + '/camera_cal/calibration*.jpg') #this loads all the calibration images

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# arrays to store object points
objpoints = [] #3D points in real world space
imgpoints = [] #2D points in image plane
ii=0
for fname in images:
    #print('reading image',ii)
    img = cv2.imread(fname)
    #fig = plt.figure(figsize=(12,15))
    #plt.imshow(img)
    # prepare object points, like (0,0,0), (1,0,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # convert image to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converts 3 channel colored image to grayscale
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    ii+=1
    if ret == True:
        #print('corners detected')
        imgpoints.append(corners)
        objpoints.append(objp)
        #fig1 = plt.figure(figsize=(12,15))
        #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        #plt.imshow(img)
        #cv2.imwrite(os.getcwd() +'/output_images/'+'detected.jpg',img) 

        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

with open(os.getcwd() + '/calibration_pickle.p','wb') as f: #this will dump all values to a pickle file
    pickle.dump(ret,f)
    pickle.dump(mtx,f)
    pickle.dump(dist,f)
    pickle.dump(rvecs,f)
    pickle.dump(tvecs,f)
'''
the following code can undistort camera calibration images [ignore this section]

'''
#for fname in images:
#    img = cv2.imread(fname)
#    undist = cv2.undistort(img, mtx, dist, None, mtx)
#    #fig = plt.figure(figsize=(12,15))
#    #plt.imshow(img)
#    #fig1 = plt.figure(figsize=(12,15))
#    #plt.imshow(undist)
#    cv2.imwrite(os.getcwd() +'/output_images/'+'undistorted.jpg',undist) 
'''
ret, mtx, dist, rvecs, tvecs - these arrays complete camera calibration
''' 
    
'''
Part 2 
'''
with open(os.getcwd() + '/calibration_pickle.p','rb') as f: #this will load all values from a pickle file
    ret = pickle.load(f)
    mtx = pickle.load(f)
    dist = pickle.load(f)
    rvecs = pickle.load(f)
    tvecs = pickle.load(f)

'''
[ignore this section]

#img = cv2.imread(os.getcwd() +'/camera_cal/calibration1.jpg')
#undist = cv2.undistort(img, mtx, dist, None, mtx)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 15))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=15)
#ax2.imshow(undist)
#ax2.set_title('Undistorted Image', fontsize=15)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''
#This works fine
'''
# Step 2.1 Color Space Change
############################################### funciton to laod test images and undistort them //
#input_img = cv2.imread(os.getcwd() +'/test_images/test5.jpg')
#undist = cv2.undistort(input_img, mtx, dist, None, mtx)
#cv2.imwrite(os.getcwd() +'/test_images/undistorted_test5.jpg',undist)
###############################################
#test_img = mpimg.imread(os.getcwd() +'/test_images/undistorted_test5.jpg')
################################################
'''
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return grad_binary
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return mag_binary
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return dir_binary
def color_threshold(img, s_thresh=(0, 255), v_thresh=(0, 255)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_binary == 1) & (v_binary == 1)] = 1
    return color_binary
def region_of_interest(img, vertices): #this was not used in advanced lane finding project
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#input_imgs = glob.glob(os.getcwd() + '/test_images/test*.jpg')
#input_imgs = glob.glob(os.getcwd() + '/test_images/straight_lines*.jpg')
def window_mask(width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output
    
def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids
''' the pipeline begins here '''
def process_image(fname):
    #img = cv2.imread(fname)
    img = fname
    #image_name=os.path.split(fname)[1]
    undist = cv2.undistort(img, mtx, dist, None, mtx) #perform distortion correction
    cv2.imwrite(os.getcwd() +'/output_images/'+'undist.jpg',undist) 
    ksize = 13 # Choose a larger odd number to smooth gradient measurements
    inp_img = cv2.imread(os.getcwd() + '/output_images/undist.jpg')

    ##############################################################################################
    '''
    Thresholding functionality 
    '''
    ##############################################################################################
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(inp_img, orient='x', sobel_kernel=ksize, thresh=(100, 100))
    grady = abs_sobel_thresh(inp_img, orient='y', sobel_kernel=ksize, thresh=(85, 100)) #85,100
    mag_binary = mag_thresh(inp_img, sobel_kernel=ksize, mag_thresh=(85, 150))
    dir_binary = dir_threshold(inp_img, sobel_kernel=ksize, thresh=(0.7, 1.4))
    color_binary = color_threshold(inp_img, s_thresh=(95, 255), v_thresh=(200, 255))#130,255 100,255
    ##############################################################################################
    # combining all the thresholds
    combined = np.zeros_like(inp_img)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255
    #combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255
    cv2.imwrite(os.getcwd() +'/output_images/'+'thresh_F.jpg',combined) 
    #combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255
    # Plot the result
    inp_img = cv2.imread(os.getcwd() + '/output_images/thresh_F.jpg')
    #vertices = np.array([[(200,682),(550, 464), (775,464), (1200,682)]], dtype=np.int32) #was 725
    #masked_edges = region_of_interest(inp_img, vertices)
    #cv2.imwrite(os.getcwd() +'/output_images/'+'masked.jpg',masked_edges) 
    #inp_img = inp_img[:,:,::-1]
    #masked_edges=masked_edges[:,:,::-1]
    masked_edges = inp_img
    img_size = (inp_img.shape[1], inp_img.shape[0])
    src = np.float32([[570,464],[710,464],[1084,682],[225,682]])
    offset = 200 #300 wroks 
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binary_warped = cv2.warpPerspective(inp_img, M, img_size)
    cv2.imwrite(os.getcwd() +'/output_images/'+'warped.jpg',binary_warped) 
    binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    # window settings
    window_width = 40 
    window_height = 144 # Break image into 9 vertical layers since image height is 720
    margin = 35 # How much to slide left and right for searching 25 works
    
    window_centroids = find_window_centroids(binary_warped, window_width, window_height, margin)
    
    # If we found any window centers
    if len(window_centroids) > 0:
    
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8)
    #fig = plt.figure(figsize=(12,15))
    
    # Display the final results
    #plt.imshow(output)
    #plt.title('window fitting results' + str(ii))
    cv2.imwrite(os.getcwd() +'/output_images/'+'output.jpg',output) 
    ploty = np.linspace(0, 719, num=len(window_centroids))
    leftx=[]
    rightx=[]
    if len(window_centroids) > 0:
        for level in range(0,len(window_centroids)):
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])
    # Plot up the fake data
    # Fit a second order polynomial to pixel positions in each fake lane line
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y 
    
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    ###
    lft_line = np.column_stack((left_fitx,ploty))
    rt_line = np.column_stack((right_fitx,ploty))
    #cv2.circle(output,tuple(leftx),1,(255,0,0))
    for index in range(0,len(ploty)):
        #print(index)
        cv2.circle(output,(int(leftx[index]),int(ploty[index])),10,(255,0,0),-1)
        cv2.circle(output,(int(rightx[index]),int(ploty[index])),10,(0,0,255),-1)
    cv2.imwrite(os.getcwd() +'/output_images/'+'output_1.jpg',output) 
    cv2.polylines(output,np.int32([lft_line]),False,(255,0,0),15)
    cv2.polylines(output,np.int32([rt_line]),False,(0,0,255),15)
    for index in range(0,len(ploty)):
        #print(index)
        cv2.circle(output,(int(leftx[index]),int(ploty[index])),10,(255,255,255))
        cv2.circle(output,(int(rightx[index]),int(ploty[index])),10,(0,0,0))
    cv2.imwrite(os.getcwd() +'/output_images/'+'output_2.jpg',output) 
    ###
    y_eval = np.max(ploty)/2
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, np.asarray(leftx)*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, np.asarray(rightx)*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp_2=np.copy(color_warp)
    color_warp_3=np.copy(color_warp)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the binary_warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp_2, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,0,255), thickness=15)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    newwarp2 = cv2.warpPerspective(color_warp_2, Minv, img_size) 

    # Combine the result with the original image
    #newwarp=newwarp[:,:,::-1]
    result = cv2.addWeighted(img, 1, newwarp, 0.4, 0)
    result = cv2.addWeighted(result, 1, newwarp2, 1, 0)
    debug_1 = cv2.addWeighted(masked_edges, 1, newwarp, 0.3, 0)
    cv2.imwrite(os.getcwd() +'/output_images/'+'debug_1.jpg',debug_1) 
    #cv2.imwrite(os.getcwd() +'/output_images/'+'output_A'+ image_name[-9:],result)
    '''
    printing relevant stuff
    '''
    height = binary_warped.shape[0]
    car_position = binary_warped.shape[1]/2
    l_fit_x_int = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2] #
    r_fit_x_int = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center_dist = (car_position - lane_center_position) * xm_per_pix
    #cv2.rectangle(result,(80,0),(600,150),(255,255,255),-1)
    rect_pt = np.array([[0,0],[0,150],[525,150],[525,0]], np.int32)
    rect_pt = rect_pt.reshape((-1,1,2))
    cv2.fillPoly(color_warp_3, np.int_([rect_pt]), (255,255,255))
    result = cv2.addWeighted(result, 1, color_warp_3, 0.15, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if left_curverad > 2500:
        text = '~Straight Road'
    else:
        text = 'Curve radius: ' + '{:04.2f}'.format(left_curverad) + 'm'
    cv2.putText(result, text, (20,50), font, 1.2, (0,0,0), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    text = '{:04.3f}'.format(abs(center_dist)) + 'm ' + direction + ' of center'
    cv2.putText(result, text, (20,125), font, 1.2, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(os.getcwd() +'/output_images/'+'resultan.jpg',result[:,:,::-1]) 
    return result
#'''
#test on images  
#'''
#import os
#path=os.listdir("test_images/")
#i=1
#for filename in path:
#    if fnmatch.fnmatch(filename, '*.jpg'):
#        image = cv2.imread("test_images/"+filename)
#        lane_det=process_image(image)
#        cv2.imwrite(os.getcwd() +'/output_images/'+filename[:-4]+'_processed.jpeg',lane_det)
#        i=i+1

#Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
white_output = 'project_video_proc.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)