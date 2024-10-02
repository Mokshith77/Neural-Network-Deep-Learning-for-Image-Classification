Project: Neural Network & Deep Learning for Image Classification
Overview: This project focuses on the implementation of a neural network model to classify images by transforming them into non-overlapping patches and using a multi-layer perceptron (MLP) architecture for classification. The project demonstrates how different optimizers (SGD and Adam) impact the model’s performance, and analyzes the training and test accuracies, loss evolution, and hyperparameter tuning.

Key Features:
1. Image Processing:
Image Patching: We used Einops to convert images into non-overlapping patches, transforming each patch into a feature vector, which is stored in a matrix for further processing.
2. Model Architecture:
Multi-layer Perceptron (MLP): The backbone of the model consists of two MLP layers where the output of one layer is fed into the next.
Activation Function: The ReLU activation function was used for its efficiency and ability to avoid the vanishing gradient problem by selectively activating neurons.
3. Classifier:
Linear Classifier: A simple linear classifier was employed to map the processed features into classes. The output is constrained to a certain range (e.g., between 0 and 1 or -1 and 1), enabling the generation of decision boundaries for classification.
4. Loss Function:
Cross-Entropy Loss: Used to calculate the error and likelihood of the model for each data point. This loss function is well-suited for classification tasks as it penalizes incorrect predictions more heavily.
5. Optimizers:
SGD (Stochastic Gradient Descent): The model initially used SGD, resulting in an accuracy of 83.4%. However, the computation time was longer compared to Adam.
Adam Optimizer: Adam was used to achieve better results by combining RMSProp and AdaGrad, with an accuracy improvement to 84%. The computation time was also faster, making it a more efficient choice for this task.
6. Hyperparameters:
Epochs: The model was trained for 50 epochs, and the accuracy increased with more epochs, reaching 86.6%.
Learning Rate: A learning rate of 0.01 was selected for the Adam optimizer, striking a balance between speed and stability in training.
Weight Decay: Set to 0 for this project.
Results:
Training Accuracy: The final model achieved a training accuracy of 86.6%.
Test Accuracy: The test accuracy was slightly lower than the training accuracy, indicating some overfitting but maintaining competitive performance.
Loss Curve: The loss decreased steadily across epochs, showing a successful optimization process.
Optimizer Comparison: Adam proved to be more efficient than SGD in terms of both accuracy and computation time.
Conclusion:
This project demonstrates the implementation of a deep learning pipeline for image classification, using MLP layers, ReLU activation, and optimizers to refine the model’s performance. With Adam optimizer and cross-entropy loss, the model achieved a solid accuracy of 86.6% on the dataset. The project also explores how optimizer choice and hyperparameter tuning impact overall model accuracy and training time.

