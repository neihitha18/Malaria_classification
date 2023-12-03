# Malaria_classification
This repository is dedicated to the classification of malaria cells into parasitized and non-parasitized categories using cutting-edge supervised machine learning and deep learning techniques. 

# Key Features:
> Machine Learning Techniques: supervised machine learning methodologies like Multi-Layer Perceptron(MLP) and Support Vector Machine (SVM) and deep learning methodologies like Convolution Neural Network (CNN) are employed to enhance the accuracy of our malaria cell classification models.

> Model Comparison: The core of this project involves comparing the results obtained from different classification models. Through this comparative analysis, the most effective model for accurately distinguishing between parasitized and non-parasitized malaria cells within image data is identified.

> Metrics Evaluation: The evaluation process is extensive by incorporating a range of metrics to provide a nuanced understanding of each classification model's performance. This includes accuracy, precision, sensitivity, specificity, F1-score, and matthews correlation constant.

# Data Analysis
Dataset used : https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
The malaria cells dataset consists of parasitized and uninfected cell images in two separate folders. There are 27,558 number of total images and 13778 uninfected cell images and 13780 infected/ parasitized cell images.

Observation: The infected cells has smaller patches present throughout the cell. While the uninfected cells does not contain and patches.  

![image](https://github.com/neihitha18/Malaria_classification/assets/60841944/82d814a1-0afe-4354-af78-1d241d78e8d4)

This will be a significant feature in classifying the parasitized cell images from healthy cell images. Although few cell images labelled as uninfected also contains tiny patches similar to parasitized cell. This can confuse the model in predicting the healthy cell as infected. 

# Data Pre-processing

 The parasitized and uninfected data are resized and labelled as 0 for parasitized and 1 for uninfected individually while reading the images. The images and labels are stored in two different variables and are converted into a numpy array. 
 The images are usually 2 dimensional with 1 or 3 colour channels. Here, the shape of the ima ge data is (27558, 150, 200, 3). 
 The image data is normalized and converted into float in ord er to scale the values between 0 to 1 by dividing each pixel with 255(The maximum value of each pixel). 
 Since the label data is integer value one-hot encoding is performed to convert th e data to categorical value using to_categorical function from tensorflow.keras. 
Once the image data is converted into array. 
 The data is split into testing and training data in 80:20 ratio respectively using the function train_test_split from sklearn.model_selection.  

# Model building and Comparision

A Multi-Layer Perceptron, Support Vector Machine and Convolutional Neural networks models are built, trained and tested against the test data.
All three models are evaluated and the results are compared as below.

<img width="459" alt="image" src="https://github.com/neihitha18/Malaria_classification/assets/60841944/e9aa850b-ebc0-43cb-b3db-a9977b7d1b68">

Matthew’s correlation coefficient is a single value that summarizes the entire confusion matrix. The values near +1 indicates best agreement between predicted and actual values and the values near 0 indicates no agreement and prediction is random. 

From the above table, it is clear that the predictions made by Convolution Neural Network model is much better than the other two models as the accuracy in prediction of CNN is 0.95 whereas for the other two models the accuracy is 0.5.

# Findings

	The accuracy of the CNN model is 0.95 when only four convolution layers are used. 
When the number of convolution layers are increased higher accuracy can be achieved 
	The implementation of larger datasets is much easier compared to other models though the training is a time consuming process. 
	The SVM model can handle large dimensional inputs which is not in the case of multilayer perceptron. 
	The performance of the SVM can be improved by adding more number of layers and applying regularization techniques like Dropout, maxpooling 
	The input of multi-layer perceptron is a giant one-dimensional vector which results in too many parameters to train. The performance of MLP with smaller datasets will result in good predictions. 

# Conclusion

In conclusion, The comparative analysis of classification models for malaria cell images underscores the remarkable efficacy of the Convolutional Neural Network (CNN) model. The results presented in the table unmistakably demonstrate the superiority of the CNN model over the alternative models, namely Support Vector Machine (SVM) and Multi-Layer Perceptron (MLP).

With an impressive accuracy rate of 0.95, the CNN model outshines its counterparts, which achieved an accuracy of 0.5. This substantial discrepancy in accuracy emphasizes the CNN model's exceptional ability to make precise and reliable predictions in the classification of parasitized and non-parasitized malaria cells.
