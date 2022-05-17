# What Dog?
## Using Convolutional Neural Networks to Classify Images by Dog Breed
### A Capstone Project for the Udacity Data Scientist Nanodegree
## Project Overview <a name="overview"></a>
### Welcome to the Project

This is the framework and codebase for a web-hosted application that will accept an image, and use a pre-trained algorithm to attempt to identify whether the image features a dog or not, and then attempt to classify said image into one of 133 dog breeds. If you merely want to run the web app, jump to the [Application](#application) section.

I chose this project because the concept of Machine Vision is fascinating to me and I believe will continue to improve and be an extremely poweful tool in all sorts of applications, of which the most popular right now is probably the self-driving car, but I plan in the future to take this into the realm of Environmental Conservation, with a special interest in public education and remote-sensing applications.

What Dog is an exercise in Convolutional Neural Networks (CNNs), and stands on the shoulders of giants: First using OpenCV's implementation of [Haar feature-based cascade classifiers](https://github.com/opencv/opencv/tree/master/data/haarcascades) as well as [ResNet50](https://arxiv.org/abs/1512.03385) to detect a dog or human face in an image, it then uses a CNN using Keras and Tensorflow with Simonyan & Zisserman's [VGG-19 Convolutional Network Model](https://arxiv.org/abs/1409.1556), Szegedy et al.'s [Inception V3](https://arxiv.org/abs/1512.00567), and Chollet's [Xception](https://arxiv.org/abs/1610.02357)(which were trained on the [ImageNet Image Database](https://www.image-net.org/)) as a base to attempt to classify the image into a dog breed (To see which dog breeds, check dog_breeds.py).

The model was trained on a base of 8351 preclassified images of dogs, provided as part of the Udacity Data Scientist Nanodegree. The majority of the codebase of this project comes from the original [Udacity guide.](https://github.com/udacity/dog-project)

### Problem Statement

To make this an effective application of CNNs, the algorithm needs to do two things: 

1. Correctly identify the presence of a dog in an image.
2. Correctly classify an image of a dog into its appropriate breed.

Correctly identifying the presence of a dog in an image is a binary classification that is reasonably easy and allows us to use previously trained models of large image sets such as ImageNet, which is conveniently classified into several labels related to dogs. ResNet50 is a previously trained model that should perform well in this case.

Correctly classifying an image of a dog into separate dog breeds is a much more complicated task that even humans might struggle with, for example: What is the difference between a Whippet and an Italian Greyhound? Simlarly, there are several colors of labrador retrievers, how would the model know to ignore color and still classify them as the same breed?

Given 133 dog breeds, assuming a perfectly balanced dataset, randomly assigning a breed to a random image of a dog can be assumed to have an accuracy of 1/133 or less than 0.8%. However, CNNs are very powerful and currently existing models such as ResNet50 have obtained over 95% accuracy on more complicated datasets. Given that this is a project with limited time and resources for training and tuning hyperparameters, it is unlikely that this model will achieve such high numbers, but should be able to obtain results significantly better than random chance.

### Metrics

The model uses two metrics: A loss function for recursive optimization of its own features, and an evaluation metric for scoring the performance of the final model at hand.

Loss is the most important metric here, as a function of how well the model performs on the training dataset vs the validation dataset. A function with high loss between training and validation cannot be said to be a good model, and is  likely over-tuned to the training data. Keras offers many loss functions by which to adjust a CNN model, and since this is a classification problem, Categorical Crossentropy was chosen as the metric by which to measure loss as opposed to Regression loss functions such as Mean Squared Error or Cosine Similarity. Categorical Crossentropy is not the only suitable loss function for this problem, but in the interest of time, it was not compared to the others, and may not be the most efficient.

For the evaluation metric, Accuracy is sufficient. Since we are mostly interested in the number of correctly classified dog breeds, the total correct divided by the total is appropriate. In the case that the incorrectly classified dog pictures became relevant in some use case, it would make sense to try to maximize precision or F-score depending on the circumstances.

## Installation <a name="installation"></a>

To reproduce the project in its entirety, perform the following steps:

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)** follow this step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.8
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.8
	activate dog-project
	pip install -r requirements/requirements.txt

7. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
8. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

9. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook and follow its contents in sequence.
```
jupyter notebook dog_app.ipynb
```

## Data Analysis <a name="analysis"></a>
### Data Exploration

The dataset of 8351 images of dogs has been split ahead of time in to folders of 6680 training, 836 testing, and 835 validation images. Within each folder is contained a folder of each breed into which all images of said breed are included. The images themselves are of varying sizes and resolutions, with the dog generally being the subject of the image, located in the center, sometimes the images are portrait-style, featuring mainly the face, and other times the whole body is featured. The majority of images feature a frontal view of the dog's face, and will sometimes feature multiple dogs, other humans present, and a wide variety of backgrounds and lighting conditions. The smallest category contains 26 images and the largest contains 55, with an average of 36.2 images per breed.

### Data Visualization

[vis1]: ./images/data_vis1.png

A quick bar plot of the training dataset shows that while the data is not perfectly distributed, it is reasonably even, and we most likely do not have to worry about issues with data imbalance.

![Available Training Images][vis1]

## Methodology <a name="methodology"></a>
### Data Preprocessing

Images were first standardized into a target size of 225*225 pixels using Keras.preprocessing.image.load_img() with default arguments, and then converted to 4D tensors of size (1, 224, 224, 3) using Keras.image.img_to_array() with default arguments.

### Implementation and Refinement

1. Human Detection

Using the cv2 library by OpenCV and their frontal face Haar feature-based cascade classifier, the function face_detector() correctly identifies faces in pictures approximately 99% of the time. It does however find human faces in pictures dogs approximately 12% of the time.

2. Dog Detection

Using the Keras library's implementation of the ResNet50 algorithm, the function dog_detector() correctly identified the presence of dogs in 100% of test images, and with 0 false positives in pictures of human beings.

3. A CNN from Scratch to Classify Dog Breeds

Using the Keras library, a CNN model was created using Convolutional 2D layers and MaxPooling2D layers, with a final GlobalAveragePooling2D followed by a Dense classification layer of 133 categories to examine how very basic changes would affect the classification task. The model was compiled with an adadelta optimizer after selecting various options, and was trained for 20 epochs. Adadelta proved to be faster but didn't appear to increase the loss of the algorithm. Beginning with 4 layers and a convolutional layer with 16 filters performed roughly as well as chance. 2 more Conv2D and MaxPooling2D layers were added with increasing number of filters. Each addition improved the model's accuracy and loss until the model had the following architecture, which had an accuracy of better than 1%:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 224, 224, 16)      208       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 112, 112, 16)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 112, 112, 32)      2080      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 56, 56, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 56, 56, 64)        8256      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 28, 28, 64)       0         
 2D)                                                             
                                                                 
 global_average_pooling2d (G  (None, 64)               0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 133)               8645      
                                                                 
=================================================================
Total params: 19,189
Trainable params: 19,189
Non-trainable params: 0
_________________________________________________________________

At this point the model began to run into memory issues with the Nvidia RTX 3080 GPU that was being used for the model, and so in the interest of time, I moved onto the next section. This memory issue could have likely been attenuated by using generators to generate training data procedurally, and would have been a good implementation for data augmentation, but this was not implemented.

4. Using Transfer Learning to Classify Dog Breeds

A rudimentary model was built extracting the bottleneck features of Simonyan & Zisserman's [VGG-16 Convolutional Network Model](https://arxiv.org/abs/1409.1556), with a GlobalAveragePooling2D layer and Dense layer with 133 categories, much like the CNN in the previous step. This model was compiled with an rmsprop optimizer, and achieved a test accuracy of 69.9761% with a validation loss of 1.74, a vast improvement on the previous CNN.

The same step was attempted for each of the Neural Networks that are available in Keras: VGG-19, ResNet50, Inception, and Xception. 

The best model was then taken and repeated with different hyperparameters, particularly the loss function and optimizer, adjusting the number of epochs up to 100 depending on how long it took the model to stabilize.

## Results <a name="results"></a>

With the exception of ResNet50, which had cumbersome issues with data input size and was thus skipped, each model proved better than the last, as seen in the table below:

| Model | Validation Loss | Accuracy |
|-------|-----------------|----------|
| VGG-16 | 1.74896 | 69.9761% |
| VGG-19 | 1.78596 | 73.4450% |
| Inception | 0.65107 | 77.3923% |
| Xception | 0.47962 | 84.4498% |

Xception, being the highest performing model, was then tested under a manual, non-exhaustive gridsearch of hyperparameters as follows:

| Loss Function | Optimizer | Validation Loss | Accuracy | No. Epochs |
|---------------|-----------|-----------------|----------|
| Categorical Crossentropy | RMSprop | 0.47962 | 84.4498% | 4 |
| Categorical Crossentropy | SGD | 0.42157 | 85.2871% | 93 |
| Categorical Crossentropy | Adam | 0.44831 | 85.2871% | 4 |
| Categorical Crossentropy | Adadelta | 02.45106 | 66.0287% | >100 |
| Poisson | RMSprop | 0.01105 | 84.3301% | 5 |
| Poisson | SGD | 0.03178 | 59.2105% | >100 |
| Poisson | Adam | 0.01087 | 85.6459% | 5 |
| KLDivergence | RMSprop | 0.47693 | 85.2871% | 4 |
| KLDivergence | SGD | 0.42238 | 85.7656 | 94 |
| KLDivergence | Adam | 0.44984 | 84.9282 | 5 |

The number of epochs should be considered as only an example, since this can vary depending on many factors. Since Adadelta took so long to optimize the first time, it was omitted for the remaining loss functions in the interest of time. In the end, I selected Categorical Crossentropy with an Adam optimizer, but any of the combinations above seem to work roughly equivalently.


### Justification

Since these models are applied more or less as-is, I cannot take credit for the high levels of accuracy on display, and these are more or less directly a result of the outstanding work of the original model developers. Each one of these models stopped improving their validation metrics very quickly, from 3-7 epochs into the process, and the use of different optimizers seems to yield some improvement, but it's hard to say if some element of random chance was involved in these very small improvements. Some optimizers were very inefficient, like SGD and especially Adadelta, and while the results obtained might be comparable or even better, computational efficiency is probably more important than the small increases in accuracy for this use case.

Surprisingly, the poisson and KLDivergence metrics yielded a very low validation loss despite classification not being their intended use. Being no statistical genius myself, I'm inclined to discard them as useful metrics without further investigation, but it's interesting that the models perform equally well with these loss functions.

Regardless, the Xception model seems to work with approximately 85% accuracy regardles of the hyperparameters used, which is very good, and with a lower loss than any of the other functions.

## Conclusion <a name="conclusion"></a>
### Reflection

Looking back at the whole process, the only conclusion I can draw is how powerful the available tools are for Deep Learning that can be generalized across all sorts of problems with surprisingly accurate results. However, I think I've only scratched the surface when it comes to optimizing CNNs, and I still have a lot of issues visualizing the process internally, particularly when it comes to the input and output needs of the various Convolutional Layers.

### Improvement

At this point, without going out of my way to specifically train a CNN for this exact application, the only improvements I can see worth making would be to augment the image data, altering variables like skew, rotation, and brightness of an image. Additionally, I'd be interested in seeing the effects of throwing in pictures of dogs of no particular breed to see if that would confound or improve the model. In application, the model appears to perform much better with pictures of a dog either in profile or facing the camera, so adding pictures of dogs from above might also be useful in improving the model.

## Application<a name="application"></a>
### How to Run

Navigate to the app folder and run main.py
Upload your chosen image when prompted, and see the results.
