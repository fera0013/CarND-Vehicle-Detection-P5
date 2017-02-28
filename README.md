**Vehicle Detection Project**
---

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[hog_hls_0]: ./output_images/hog_features/hls_0.png 
[hog_hls_2]: ./output_images/hog_features/hls_2.png
[hog_hsv_0]: ./output_images/hog_features/hsv_0.png
[positives1]: ./output_images/positives/test1.png
[positives2]: ./output_images/positives/test2.png
[positives3]: ./output_images/positives/test3.png
[positives4]: ./output_images/positives/test4.png
[positives5]: ./output_images/positives/test5.png
[positives6]: ./output_images/positives/test6.png
[final1]: ./output_images/final/test1.png
[final2]: ./output_images/final/test2.png
[final3]: ./output_images/final/test3.png
[final4]: ./output_images/final/test4.png
[final5]: ./output_images/final/test5.png
[final6]: ./output_images/final/test6.png
[grid]: ./output_images/detection_grid/test1.png
[heat1]: ./output_images/heat_maps/test1.png
[heat2]: ./output_images/heat_maps/test2.png
[heat3]: ./output_images/heat_maps/test3.png
[heat4]: ./output_images/heat_maps/test4.png
[heat5]: ./output_images/heat_maps/test5.png
[heat6]: ./output_images/heat_maps/test6.png
[video1]: ./project_video_processed.mp4

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored the hog features for different `color_space` and  `channel` parameters, to get an idea of which color space / channel combination may yield the best results in "hog-space". The following 3 images shows the results for the three best color space / channel
combinations:

![alt text][hog_hls_0]

*Hog features for channel 0 in HLS - color space*

![alt text][hog_hls_2]

*Hog features for channel 2 in HLS - color space*

![alt text][hog_hsv_0]

*Hog features for channel 0 in HSV - color space*

For the other hog parameters, I used the default values `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Finetuning of these parameters is deferred to a future iteration of this project. 

####2. Explain how you settled on your final choice of HOG parameters.

The hog parameters were chosen based on the final detection results on the image in the `.\test_images` subfolder, which could be achieved with the hog-features of channel 0 in HLS color space. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog-features, spatial and color-histogramm features and got an accuracy of 0.97. The implementation of a *sklearn pipeline* which also includes some kind of dimensionality reduction algorithm such as *Principal Component Analysis*, is deferred to a future iteration of this project. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For sliding window search, I used a simple grid consisting of quadratic windows of size 128 and an overlap of 0.75. This grid proved to be succesful on all but one test images and the more sophisticated grid patterns I tried did not yield better results. To reduce noise, I reduced the size of the grid pattern to the region of interest. The following image illustrates the grid pattern drawn on one of the test images:

![alt text][grid]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on the grid described above using HLS channel 0 HOG features plus spatially binned color and histograms of color in the feature vector, which provided the best results on the test images. The following pictures show the detected positives on the test images:

![alt text][positives1]

![alt text][positives2]

![alt text][positives3]

![alt text][positives4]

![alt text][positives5]

![alt text][positives6]

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_processed.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

### Here are six frames and their corresponding heatmaps:

![alt text][heat1]

![alt text][heat2]

![alt text][heat3]

![alt text][heat4]

![alt text][heat5]

![alt text][heat6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][final1]

![alt text][final2]

![alt text][final3]

![alt text][final4]

![alt text][final5]

![alt text][final6]


###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A general problem in this project is the large set of hyperparameters, that makes it impossible to find the best combination using a systematic grid search approach. The only possible approach is therefore to find a promising setup by trial and error and refine it further by finetuning additional parameters. The approach I chose for this project yielded good results on the test images and is able to detect the cars most of the times in the video. The detection boxes are however extremely unstable and there are a lot of false positives. The former problem could be tackled by an algorithm that averages boxes over several frames. A solution for the false positives could be some sort of continuity tracking, which ignores boxes that are not confirmed by similar ones in subsequent frames. Since most false positives occur at the bottom of the frames, they could also be filtered out by reducing the sliding window grid further. 
