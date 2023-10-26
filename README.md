# Improving-Generalization-of-Deep-AUC-Maximization-for-Medical-Images
## Summary
Implemented multiple ResNet18 models to improve the generalization ability of Deep AUC Maximization for 7 large distinct medical image classification datasets having 2D and 3D images

## Objective
This project aims to enhance the generalization ability of classification models for various medical images data from the Medmnist dataset. The dataset comprises of 7 large distinct medical image datasets relating to various medical conditions. The Convolutional Neural Network (CNN) model with Resnet-18 architecture is used to train these images and classify the medical conditions.

## Approaches & Techniques
For 2D Datasets:
1. ### Normalization:
   Normalization is a technique used in data preprocessing to transform numerical data into a standardized scale. It is used to bring all the data points onto a common scale so that they can be compared against each other. It helps to avoid bias and improves the convergence of optimization algorithms with increasing efficiency. So we normalized the dataset. We then define the data transforms to preprocess the images and create dataloaders for the training, validation, and test datasets.
3. ### Optimizer:
   PESG (Projection Ensemble Stochastic Gradient Descent) is an optimizer that can be used to maximize the area under the ROC curve (AUC) for binary classification tasks. PESG improves upon SGD (Stochastic Gradient Descent) by using a projection ensemble method. We tried to use Adam Optimer with different parameters such as learning rate, momentum setting one and change the other but we didn’t get the intended accuracy and shifted to PESG. We started to use PESG and we got promising results and experimented with changing all the parameters one by one such as learning rate, epoch_decay, and weight decay. We started by giving lower learning rates. However, our model took time to converge and the performance was not improving with the decreasing learning rate we concluded that it’s better to start with a relatively high value of 0.1 and then lower down later on using the decay_epoch.
5. ### Hyperparameter tuning:
   Hyperparameter tuning is a critical step in machine learning that involves finding the best combination of hyperparameters for a given model to optimize its performance. As we discuss regarding the learning rate and epoch decay above, we penalize it after a certain epoch. We fined-tunned our model changing all the parameters and we got a good performance with margin = 1.0, and momentum = 0.8 after several experiments.
6. ### Early Stopping:
   Early stopping is a technique used in machine learning to prevent the overfitting of a model during training. We ran for 100 epochs and we kept the early_stop epoch as 15 which helped in getting good AUC scores before running all the remaining epochs.

For 3D datasets:
1. ### Data Augmentation:
   This technique introduces additional variations to the input data during training, which can help to reduce overfitting and improve the generalization performance of the model. We created a class “Transform3D” to apply transformations for both the training and evaluation images. This was done to randomly scale the input voxel data during training. This technique had a significant impact on the performance

## PneumoniaMNIST
Description: The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is binary-class classification of pneumonia against normal. We split the source training set with a ratio of 9:1 into training and validation set and use its source validation set as the test set

Hyperparameter Tuning Parameters:
1. Loss function: AUCM loss
2. Momentum: 0.8
3. Margin: 1.0
4. Learning rate: 0.1
5. Weight_decay: 0.001

## NoduleMNIST3D
Description: The NoduleMNIST3D is based on the LIDC-IDRI, a large public lung nodule dataset, containing images from thoracic CT scans. The dataset is designed for both lung nodule segmentation and 5-level malignancy classification task

Hyperparameter Tuning Parameters:
1. Loss function: AUCM loss
2. Momentum: 0.7
3. Margin: 1.0
4. Epoch_decay: 2e-3
5. Learning rate: 0.1
6. Weight_decay: 1e-4

## AdrenalMNIST3D
Description: The AdrenalMNIST3D is a new 3D shape classification dataset, consisting of shape masks from 1,584 left and right adrenal glands (i.e., 792 patients). Collected from Zhongshan Hospital Affiliated to Fudan University, each 3D shape of adrenal gland is annotated by an expert endocrinologist using abdominal computed tomography (CT), together with a binary classification label of normal adrenal gland or adrenal mass.

Hyperparameter Tuning Parameters:
1. Loss function: AUCM loss
2. Momentum: 0.7
3. Margin: 1.0
4. Epoch_decay: 0.05
5. Learning rate: 0.1
6. Weight_decay: 1e-4

## VesselMNIST3D
Description: The VesselMNIST3D is based on an open-access 3D intracranial aneurysm dataset, IntrA, containing 103 3D models (meshes) of entire brain vessels collected by reconstructing MRA images. 1,694 healthy vessel segments and 215 aneurysm segments are generated automatically from the complete models

Hyperparameter Tuning Parameters:
1. Loss function: AUCM loss
2. Momentum: 0.7
3. Margin: 1.0
4. Epoch_decay: 2e-3
5. Learning rate: 0.1
6. Weight_decay: 1e-4

## SynapseMNIST3D
Description: The SynapseMNIST3D is a new 3D volume dataset to classify whether a synapse is excitatory or inhibitory. It uses a 3D image volume of an adult rat acquired by a multi-beam scanning electron microscope.

Hyperparameter Tuning Parameters:
1. Loss function: AUCM loss
2. Momentum: 0.7
3. Margin: 1.0
4. Epoch_decay: 2e-3
5. Learning rate: 0.1
6. Weight_decay: 1e-4

## Results
Dataset Test | AUC
1. PneumoniaMNIST | 0.91
2. NoduleMNIST3D | 0.89
3. AdrenalMNIST3D | 0.80
4. VesselMNIST3D | 0.86
5. SynapseMNIST3D | 0.74
