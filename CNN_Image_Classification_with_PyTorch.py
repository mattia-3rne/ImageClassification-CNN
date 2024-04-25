from keras.datasets import fashion_mnist
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()




# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Initialize an array to keep track of which class indices have been displayed
class_indices_displayed = np.zeros(len(class_names), dtype=bool)

# Display one sample image from each class along with label and index
plt.figure(figsize=(12, 6))
for i in range(len(trainY)):
    label_index = trainY[i]
    if not class_indices_displayed[label_index]:
        plt.subplot(2, 5, label_index + 1)
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.grid(False)  # Disable grid lines
        plt.imshow(trainX[i], cmap=plt.cm.binary)  # Display image
        plt.xlabel(f"{class_names[label_index]} ({label_index})")  # Set label for x-axis
        class_indices_displayed[label_index] = True  # Mark class index as displayed
    if np.all(class_indices_displayed):  # Break loop if all classes have been displayed
        break
plt.tight_layout()
plt.show()




train_x, val_x, train_y, val_y = train_test_split(trainX, trainY, test_size = 0.1)

# converting training images into torch format
train_x = train_x.reshape(54000, 1, 28, 28)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# converting validation images into torch format
val_x = val_x.reshape(6000, 1, 28, 28)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)




## Architecture

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=2, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=2, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    



    # define the model
model = Net()
# define the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

## function to train the model

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    x_train = x_train.float()
    x_val = x_val.float()
    output_train = model(x_train)
    output_val = model(x_val)

    # Ensure that target tensors are of type torch.LongTensor
    y_train = y_train.long()
    y_val = y_val.long()

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    #if epoch%2 == 0:
        # printing the validation loss
    print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)


#Reshaping and Conversion
test_x = testX.reshape(10000, 1, 28, 28)
test_x  = torch.from_numpy(test_x)

# generating predictions for test set
with torch.no_grad():
    test_x = test_x.float()
    output = model(test_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

test_y = testY.astype(int);
test_y = torch.from_numpy(test_y)


# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []


# Empty lists to store training, validation, and test accuracies
train_accuracies = []
val_accuracies = []
test_accuracies = []

# Training the model
for epoch in range(n_epochs):
    train(epoch)
    
    # Predictions for the training set
    with torch.no_grad():
        train_x = train_x.float()
        output = model(train_x)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    # Accuracy on the training set
    training_accuracy = accuracy_score(train_y, predictions)
    train_accuracies.append(training_accuracy)

    # Predictions for the validation set
    with torch.no_grad():
        val_x = val_x.float()
        output = model(val_x)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    # Accuracy on the validation set
    validation_accuracy = accuracy_score(val_y, predictions)
    val_accuracies.append(validation_accuracy)

    # Predictions for the test set
    with torch.no_grad():
        test_x = test_x.float()
        output = model(test_x)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    # Accuracy on the test set
    test_accuracy = accuracy_score(test_y, predictions)
    test_accuracies.append(test_accuracy)

# Convert accuracies to percentages
train_accuracies_percent = [accuracy * 100 for accuracy in train_accuracies]
val_accuracies_percent = [accuracy * 100 for accuracy in val_accuracies]
test_accuracies_percent = [accuracy * 100 for accuracy in test_accuracies]

# Plotting the accuracies
plt.plot(range(1, n_epochs+1), train_accuracies_percent, label='Training Accuracy')
plt.plot(range(1, n_epochs+1), val_accuracies_percent, label='Validation Accuracy')
plt.plot(range(1, n_epochs+1), test_accuracies_percent, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training, Validation, and Test Accuracies')
plt.legend()
plt.show()


# Predictions for the training set
with torch.no_grad():
    train_x = train_x.float()
    output = model(train_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# Accuracy on the training set
training_accuracy = accuracy_score(train_y, predictions)
training_accuracy_percent = training_accuracy * 100
print(f"Accuracy on the training set: {training_accuracy_percent:.2f}%") # Training Set is 90% of the data-set

# Predictions for the validation set
with torch.no_grad():
    val_x = val_x.float()
    output = model(val_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# Accuracy on the validation set
validation_accuracy = accuracy_score(val_y, predictions)
validation_accuracy_percent = validation_accuracy * 100
print(f"Accuracy on the validation set: {validation_accuracy_percent:.2f}%") # Validation Set is the remaining 10% of the data-set


# Predictions for the test set
with torch.no_grad():
    test_x = test_x.float()
    output = model(test_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

test_y = testY.astype(int)
test_y_tensor = torch.from_numpy(test_y)

# Accuracy on the test set
test_accuracy = accuracy_score(test_y_tensor, predictions)
test_accuracy_percent = test_accuracy * 100
print(f"Accuracy on the test set: {test_accuracy_percent:.2f}%") # Test Set is the whole 100% of the data-set




# Define a function to display sample images with predictions and actual labels
def display_sample_images(images, predictions, actual_labels, class_names, num_images=25):
    num_cols = 5  # Number of columns in the grid
    num_rows = (num_images + num_cols - 1) // num_cols  # Number of rows in the grid

    plt.figure(figsize=(15, 15))

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        # Remove the extra dimension from the image data
        image = np.squeeze(images[i])
        plt.imshow(image, cmap='gray')  # Display the image
        # Display both predicted and actual labels
        predicted_label = predictions[i]
        actual_label = actual_labels[i]
        plt.title(f"Predicted: {class_names[predicted_label]} ({predicted_label})\nActual: {class_names[actual_label]} ({actual_label})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Select a random sample of 25 images and their corresponding predictions and actual labels
num_images_to_display = 25
indices = np.random.choice(len(test_x), num_images_to_display, replace=False)
sample_images = [test_x[i] for i in indices]
sample_predictions = [predictions[i] for i in indices]
sample_actual_labels = [test_y[i] for i in indices]

# Assuming you have a list of class names like this
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display the sample of images with predictions and actual labels
display_sample_images(sample_images, sample_predictions, sample_actual_labels, class_names, num_images_to_display)