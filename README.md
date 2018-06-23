# cat-dog-image-classifier

## Complete Steps to implement a CNN to classify between cat and dog image

### Step 1 : Getting the Dataset 
    
    The dataset is available in the link : https://www.kaggle.com/c/dogs-vs-cats/data
    Download this dataset, extract and store it in localdisk
    
### Step 2 : Installing Required Packages [Python 3.6]

    1. OpenCV     ---> '3.4.0'     [ Used to handle image operations like : reading the image , resizing , reshaping]
    2. numpy      ---> '1.14.4'    [ Image that is read will be stored in an numpy array ]
    3. TensorFlow ---> '1.8.0'     [ Tensorflow is the backend for Keras ]
    4. Keras      ---> '2.1.6'     [ Keras is used to implement the CNN ]

### Step 3 : How the Model Works ??

    Note : Spyder is used to develop the code. Set the working Directory Correctely.
           Open the trained_model.py file and set the image you want to test [Ref. Line Number 24 ]
           
    This will predict the output for the image you have specified. 
    Once you have understood the basic working of the model, its now time to build the classifier from the scratch.
    
### Step 4 : Building the Classifier

    Steps involved in building a classifier is briefed in classifier.py
    13 actual substeps are involved in building your CNN! 
    This will build the model and save the trained model as a Json file. 
    The weights of the trained model are stored in a seperate file.
    
 ### Step 5: How to use the trained Model ?
 
    Once you have imported the packages, you can directely load your trained classifier from the json file.
    Then load the weights from the h5 file saved in the previous step.
    This basically has 6 substeps before you can see the result.
    
    
