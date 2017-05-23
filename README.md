**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration3.jpg "Raw Calibration Image"
[image2]: ./output_images/calibration3_corners.jpg "Detected Corners"
[image3]: ./output_images/cal3_undistorted.jpg "Undistorted Image"
[image4]: ./test_images/test2.jpg "Test Image"
[image5]: ./output_images/test2_undist.jpg "Undistorted Test Image"
[image6]: ./output_images/test2_thresh_F.jpg "Thesholded Test Image"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Part 1 of 'adv_lane_lines.py' file.
The objpoints and img point arrays are initiated as empty arrays. The calibration image file is read using ```cv2.imread``` function.
The ```cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)``` function converts a 3 channel colored image to a grayscale image. ```cv2.findChessboardCorners(gray, (nx, ny), None)``` function with correct gird parameters (in this case 6x9 corners) results into detected corners stored in an array which is then appended to imgpoints. The object points is prepared using ```np.mgrid[0:9,0:6].T.reshape(-1,2)``` which is further appended to objpoints. 

The object point and image point matrices are then used to perform calibration on the camera using the function ```cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)```. The cv2.calibrateCamera function returns distortion coefficients (dist), camera matrix (mtx), rotation and translation vectors (rvecs and tvecs).

The distortion coefficients along with camera matrix is used to undistort an image
 ```undist = cv2.undistort(img, mtx, dist, None, mtx)``` 
This returns an image (undist) which is free of any kind of distortion.

The camera is calibrated only once, the calibration coefficients are then stored in a pickle file named ```calibration_pickle.p```

| Raw Calibration Image | Detected Corners | Undistorted Image |
|:---:|:---:|:---:|
| ![alt text][image1] | ![alt text][image2] | ![alt text][image3] |

*Figures show the raw calibration image, calibration image with detected corners and the image after applying distortion correction*

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The ```cv2.undistort``` function is incorporated in the pipeline, so that distortion correction is applied to each image. 

| Raw Camera Image | Undistorted Image |
|:---:|:---:|
| ![alt text][image4] | ![alt text][image5] |

*Figures show the raw camera image and the image after applying distortion correction. The car on the corner of the raw image is not visible in undistorted image*

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

After performing distortion correction, following combinations of thresholding functions are used to minimize everything but the lane lines.
1. ```gradx = abs_sobel_thresh(inp_img, orient='x', sobel_kernel=ksize, thresh=(100, 100))``` Sobel Operator in X (line 212) 
2. ```grady = abs_sobel_thresh(inp_img, orient='y', sobel_kernel=ksize, thresh=(85, 100))``` Sobel Operator in X (line 213)
3. ```mag_binary = mag_thresh(inp_img, sobel_kernel=ksize, mag_thresh=(85, 150))``` Magnitude threshold (function mag_thresh - line 122 through 138)
4. ```dir_binary = dir_threshold(inp_img, sobel_kernel=ksize, thresh=(0.7, 1.4))``` Direction threshold (function dir_threshold - line 139 through 152)
5. ```color_binary = color_threshold(inp_img, s_thresh=(95, 255), v_thresh=(200, 255))#130,255 100,255``` Color Thresholding (function color_threshold - line 153 through 169)

The sobel X and Y operators perform derivative operation in X and Y direction. 
The magnitude thresholding is gradient of the square root of the sum of squares of the individual x and y. This type of thresholding smooths over the noisy intensity fluctuations on the small scale. [1] 
'The direction of the gradient is simply the inverse tangent (arctangent) of the y gradient divided by the x gradient' [2] The direction gradient helps in picking out the particular feature from the image (in this case, the lane lines)[2]
The color thresholding uses the HLS and HSV colorspaces for applying thresholds for a particular color value. In this case, I have used the combination of S and V color thresholding. The function ```color_threshold``` uses ```cv2.cvtColor(img, cv2.COLOR_BGR2HLS)``` and ```cv2.cvtColor(img, cv2.COLOR_BGR2HSV)``` to convert the image to HLS and HSV color spaces. Each image is then thresholded for S and V channel.  

| Undistorted Image | Thresholded Image |
|:---:|:---:|
| ![alt text][image5] | ![alt text][image6] |

*Figures show the camera image after applying distortion correction and thresholded image after applying the combined thresholds*

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

### References 
[1] Lesson 22 Magnitude of the Gradient - Project: Advanced Lane Finding 
[2] Lesson 23 Direction of the Gradient - Project: Advanced Lane Finding 
