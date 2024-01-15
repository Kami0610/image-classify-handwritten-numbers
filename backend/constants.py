import torch


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Directories and files
USER_IMAGES = '../images/'
MODEL_FILE_NAME = '../mnist_model.pth'
