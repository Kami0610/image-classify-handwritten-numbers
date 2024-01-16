import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import constants as const
from neural_network import ConvNeuralNet

# Data transformation
data_transform = transforms.Compose([
    # ±20 degrees rotation, ±10 percent resize
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

# Training dataset, which gets transformed
train_dataset = MNIST(
    root='./data', train=True, transform=data_transform, download=True
)

# Loads the training data
train_loader = DataLoader(
    dataset=train_dataset, batch_size=const.BATCH_SIZE, shuffle=True
)

# Initialize model and use chosen device
model = ConvNeuralNet().to(const.DEVICE)

# Define loss function and optimizer: Cross Entropy Loss function and Adam optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE)

# Loop through each epoch and train the model
for each_epoch in range(const.EPOCHS):
    # Set model for training
    model.train()
    # Loop through batches of training data
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = torch.tensor(data).to(const.DEVICE), target.to(const.DEVICE)

        # Zeroes out gradients
        optimizer.zero_grad()
        # Model predictions
        output = model(data)
        # Loss calculation between prediction and actual targets
        loss = loss_function(output, target)
        # Backpropagation
        loss.backward()
        # Updates the model parameters with the Adam algorithm
        optimizer.step()

        # Print progress of training
        if batch_id % 100 == 0:
            print(
                f'Epoch {each_epoch + 1}/{const.EPOCHS}, '
                f'Batch {batch_id}/{len(train_loader)}, '
                f'Loss: {loss.item()}'
            )

# Save the trained model
torch.save(model.state_dict(), const.MODEL_FILE_NAME)
