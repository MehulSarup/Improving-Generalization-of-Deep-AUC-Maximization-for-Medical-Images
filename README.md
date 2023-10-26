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


## Results
Dataset Test | AUC
1. NoduleMNIST3D | 0.89
2. AdrenalMNIST3D | 0.80
3. VesselMNIST3D | 0.86
4. SynapseMNIST3D | 0.74
