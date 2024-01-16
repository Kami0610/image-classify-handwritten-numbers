import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import constants as const
from neural_network import ConvNeuralNet

# Testing dataset
test_dataset = MNIST(
    root='./data', train=False, transform=transforms.ToTensor(), download=True
)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=const.BATCH_SIZE, shuffle=False
)

# Initialize model and use chosen device
model = ConvNeuralNet().to(const.DEVICE)

# Load the saved model for testing
model.load_state_dict(torch.load(const.MODEL_FILE_NAME))
# Set model for evaluation
model.eval()

# Accuracy counters
total_tests = 0
correct_pred = 0

# Loop through testing data, checking if model predictions match data targets. Update counters
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.clone().detach().to(const.DEVICE), target.to(const.DEVICE)
        output = model(data)
        _, prediction = torch.max(output.data, 1)
        total_tests += target.size(0)
        correct_pred += (prediction == target).sum().item()

# Calculate accuracy percentage and print value out
model_accuracy = correct_pred / total_tests
print(f'Accuracy on the test set: {model_accuracy * 100:.2f}%')
