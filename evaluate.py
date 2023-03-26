import pickle as pickle
import random
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
import config

# Load the training history dict
with open(config.HISTORY_PATH, 'rb') as f:
    model_history = pickle.load(f)

# Plot the training history
plt.figure()
plt.plot(model_history['train_loss'], label='Train loss')
plt.plot(model_history['test_loss'], label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(config.HISTORY_PLOT_PATH)

# Load the trained model
model = torch.load(config.MODEL_PATH).to(config.DEVICE)

# Load test image and mask paths
with open(config.TEST_IMGS_PATH, 'rb') as f:
    test_imgs = pickle.load(f)
with open(config.TEST_MASKS_PATH, 'rb') as f:
    test_masks = pickle.load(f)

# Get three random test images and corresponding masks
indices = random.sample(range(len(test_imgs) - 1), 3)
test_image_mask = {test_imgs[idx]: test_masks[idx] for idx in indices}

# Create a transformation composition
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor()
])

# Turn off gradient tracking
with torch.no_grad():
    # Initiate a subplot
    figure, ax = plt.subplots(3, 3, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
    # Loop over the test images
    for i, (image, mask) in enumerate(test_image_mask.items()):
        # Load the test image
        image = Image.open(image).convert('RGB')
        image = transform(image).to(config.DEVICE)
        image = torch.unsqueeze(image, dim=0)
        # Make a prediction on the image
        pred = model(image).squeeze()
        pred = torch.softmax(pred, dim=0)
        pred = torch.argmax(pred, dim=0)
        pred = pred.cpu().numpy()
        pred = pred * 255
        # Load the test mask
        mask = Image.open(mask).convert('L')
        mask = transform(mask)
        mask = torch.squeeze(mask, dim=0)
        mask = mask * 255
        mask -= 1
        # Prepare the test image for the plot
        image = image.squeeze().permute(1, 2, 0).cpu()
        # Plot the image, mask and predicted mask
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask)
        ax[i, 2].imshow(pred)

    # Save the plot
    figure.savefig(config.PRED_PLOT_PATH)
