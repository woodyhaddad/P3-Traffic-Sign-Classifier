# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_labels.jpg "Test set label quantities"
[image2]: ./training_labels.jpg "Training set label quantities"
[image3]: ./validation_labels.jpg "Validation set label quantities"
[image4]: ./grayscale_example.png
[image5]: ./German-traffic-signs/id4.png "Speed limit (70km/h) Sign"
[image6]: ./German-traffic-signs/id14.png "Stop Sign"
[image7]: ./German-traffic-signs/id25.png "Road Work Sign"
[image8]: ./German-traffic-signs/id27.png "Pedestrians Sign"
[image9]: ./German-traffic-signs/id36.png "Go Straight or Right Sign"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Link to my [project code](https://github.com/woodyhaddad/P3-Traffic-Sign-Classifier.git)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. In the code, the analysis.

I used stats.describe from the scipy library to calculate summary statistics of the traffic signs data set. I also used np.array.shape() and loops/ conditionals to find properties that were not provided by scipy's stats. It is always good to look at what scipy provides since it is an easy way to know means, variances and min/maxes, which end up giving more insight about the data we are dealing with:

* The size of training set is 34799 images
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the quantities of each label for each data set:


![alt text][image1]
![alt text][image2]
![alt text][image3]


### Design and Test a Model Architecture

#### 1. Image pre-processing:

As a first step, I decided to convert the images to grayscale because we may not need all three channels in order for our model to identify the sign. I went with quantitatively conservative datasets (since grayscale carries less information than RGB) to start with. I did test my model with RGB instead of grayscale and it did not perform better, so I stuck with grayscale. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a last step, I normalized the image data (such that data would be centered at 0) because it would make the model run more accurately.

I applied the following formula on my pixel data in order to normalize the image:

pixel = (pixel - 128.0)/128


#### 2. Model Architecture

I implemented the LeNet-5 model architecture.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation function							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| Activation function							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  		    		|
| Flatten				| 2x2 striden output 400						|
| Fully connected		| output 120									|
| RELU					| Activation function							|
| Fully connected		| output 84			    						|
| RELU					| Activation function							|
| Dropout				| Dropout function   							|
| Fully connected		| output 43			    						|





#### 3. Model Training

Here are the tensorflow functions I used in order to achieve backpropagation:

cross entropy: tf.nn.softmax_cross_entropy_with_logits
loss operation: tf.reduce_mean
optimizer: tf.train.AdamOptimizer
training operation: optimizer.minimize


I kept batch size at 128. I tried 256 but it didn't really improve the accuracy of my model.


#### 4. Model Accuracy

To find the optimal training approach and solution, I had to go through several iterations of values for my hyperparameters. I tried to isolate dependencies by changing a minimum number of values at once. But I found that the number of Epochs went hand in hand with learning rate when it came to model accuracy. For example, if I increase both by a good amount, I got overfitting (training accuracy was 99.9% and validation accuracy went down a little). I found that good values that worked well were 50 for Epochs and 0.003 for learning rate. 

I also tries several pre-processing methods that did not end up giving better results (some of which I chose to keep in my code as comments). I tried slicing the images to get rid of whatever was outside of the bounding boxes given in set['coords']. I achieved that via cv2 (see variables 'D4' under the pre-processing cell to see how I did it) 
I also tried creating more data by rotating image signs which in hindsight does not seem like a good idea since in practice, there is a very low likelihook that we will find a rotated traffic sign. I also tried eliminating the grayscaling step from my pre-processing, but my model performed better consistently with grayscaling.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.5% 
* test set accuracy of 92.4%

If an iterative approach was chosen:

My starting point for my architecture was the LeNet-5 architecture taken from the lecture. It worked well with the MNIST dataset so I decided to give it a try with traffic sign identification. After a first iteration, I noticed that my model was overfitting, so I decided to add a dropout layer in between my fully connected layers in order to reduce overfitting. A probability of 0.5 for dropout (which is typical for this kind of application) got the job done: while keeping everything else constant, my validation accuracy was improved once I added dropout with 0.5 probability.
I also had to modify input/ outputs to make the model work with this data. I had to change the 'conv1_W' weight variable to have an input depth of 1 instead of 3 (grayscale instead of RGB). Also, since we have 43 classes, I had to change my 'fc3_W' and 'fc3_b' for my last fully connected layer.

To sum up, my model at this point works pretty well since my training and validation accuracies are both well above 93%. To confirm that the model works well, I tested it on the Test set and got an accuracy of 92.4%.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Web Images

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The two hardest images to classify in my opinion are the stop sign and the pedestrians sign because the actual sign portion constitutes less of the image than it does in the other images. Also, the pedestrians sign is particularly hard to classify because the sign is not parallel to the camera lens (maybe a persspective transform would come in handy here) and the background behind the sign is not uniform.

#### 2. Model Accuracy on Web Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)	| Speed limit (70km/h)							| 
| Stop Sign    			| End of no passing								|
| Road work				| Road work										|
| Pedestrians      		| General caution				 				|
| Go straight or right	| Go straight or right 							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is lower than the accuracy on the test set of 92.4%. At a glance, this lower percentage could be due to the fact that there is little data in our training, validation and test sets about the two misclassified images (see bar charts above, images 1,2,3, labels 14 and 27). Other reasons could be any differences that arise between our images that we used and the 5 web images: using different cameras or having very different lighting conditions/ original image pixel sizes for example could mislead our CNN into looking for the wrong features and making the wrong decision.  

#### 3. Model Certainty About Web Image Prediction

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

For the first image, the model is very sure that this is a 'Speed limit (70km/h)' sign (probability of 1.0), and the image does contain a 'Speed limit (70km/h)' sign. The top probability of 1.0 contains a rounding error due to python not applying enough significant figures to the number, since if 1.0 was the true probability, all the other ones would have to be 0 by definition of Softmax. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		    | Speed limit (70km/h)						    | 
| 1.7068588e-26			| Speed limit (80km/h)							|
| 0.0000000e+00			| Speed limit (20km/h)							|
| 0.0000000e+00			| Speed limit (30km/h)			 				|
| 0.0000000e+00		    | Speed limit (50km/h) 			    			|


For the second image, the model was very sure that this is a 'Priority road' (100%) however, the correct prediction would have been "Stop" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00       		    | Priority road     						    | 
| 1.2158862e-14			| Speed limit (30km/h)                          |
| 5.9377502e-22			| Right-of-way at the next intersection         |
| 3.1331551e-22			| Speed limit (120km/h)  		 				|
| 1.8155302e-24		    | Roundabout mandatory                  		|

For the third image, the model was very certain and accurate (correct prediction at very close to 100%). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00      		    | Road work         						    | 
| 1.5043357e-31			| Bicycles crossing                             |
| 1.1699521e-31			| Wild animals crossing                         |
| 3.3121129e-37			| Bumpy road            		 				|
| 3.3995271e-38		    | Speed limit (80km/h)   	            		|

For the fourth image, the model was 100% certain that the sign was a 'Keep right' sign. The correct prediction would have been 'Pedestrians' sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9999988e-01 	    | Keep right         						    | 
| 1.5213770e-07			| Speed limit (30km/h)                          |
| 4.2433603e-09			| Right-of-way at the next intersection         |
| 1.7506623e-14			| Speed limit (60km/h)          				|
| 4.5339958e-16		    | General caution                        		|

For the fifth image, the model was very certain and accurate (correct prediction at very close to 100%). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00          	    | Go straight or right 						    | 
| 1.8505612e-17 		| Speed limit (60km/h)                          |
| 1.3485954e-17			| Road work                                     |
| 8.3097870e-18			| Yield             			 				|
| 2.2366852e-18		    | End of speed limit (80km/h)            		|
