import constants as const
import os
from PIL import Image
from neural_network import ConvNeuralNet
import torch
import torchvision.transforms as transforms


# Initialize model and use chosen device
model = ConvNeuralNet().to(const.DEVICE)

# Load the saved model for testing
model.load_state_dict(torch.load(const.MODEL_FILE_NAME))
# Set model for evaluation
model.eval()


def predict_user_image(image_path, amount=3):
    # Open image
    user_image = Image.open(image_path)
    # Resize and apply antialiasing
    user_image = user_image.resize((28, 28), Image.Resampling.LANCZOS)
    # Grayscale the image
    user_image = user_image.convert('L')
    # Transform to a tensor
    user_image = transforms.ToTensor()(user_image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        image_data = user_image.clone().detach().to(const.DEVICE)
        output = model(image_data)

    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)[0]

    # Get the top k predictions and their corresponding indices
    top_k_probabilities, top_k_indices = torch.topk(probabilities, amount)

    # Convert indices to class labels
    top_k_labels = [str(idx.item()) for idx in top_k_indices]

    # Convert probabilities to percentages
    top_k_percentages = [(prob.item() * 100) for prob in top_k_probabilities]

    # Create a dictionary with class labels and their corresponding percentages
    prediction = dict(zip(top_k_labels, top_k_percentages))

    return prediction


# Get all user-provided images
user_image_list = []
if not os.path.exists(const.USER_IMAGES):
    # If the directory does not exist, make it
    os.makedirs(const.USER_IMAGES, exist_ok=True)
else:
    # If the directory exists, get all files in the directory
    for file in os.listdir(const.USER_IMAGES):
        file_path = os.path.join(const.USER_IMAGES, file)

        # Filter for only images
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            user_image_list.append(file_path)

if not user_image_list:
    print('There were no user provided images detected. Please place images in the image folder.')
else:
    # Loops through each image provided by the user
    for each_image in user_image_list:
        # Prediction dictionary
        predicted_prob = predict_user_image(each_image, amount=3)
        # Print the image path and the prediction probabilities
        print(f'The image: {each_image}')
        for each_guess in predicted_prob:
            print(f'Prediction number: {each_guess} --> {predicted_prob[each_guess]:.2f}%')
        print('========================================')
