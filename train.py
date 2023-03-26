import os
import pickle
import time
from random import shuffle
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import config
from dataset import SegmentationDataset
from model import UNet

# Create a list of image paths
img_paths = sorted([
    os.path.join(config.IMAGE_PATH, name)
    for name in os.listdir(config.IMAGE_PATH)
    if name.endswith('.jpg')
])

# Create a list of mask paths
mask_paths = sorted([
    os.path.join(config.MASK_PATH, name)
    for name in os.listdir(config.MASK_PATH)
    if not name.startswith('.') and name.endswith('.png')
])

# Shuffle images and masks
tmp = list(zip(img_paths, mask_paths))
shuffle(tmp)
img_paths, mask_paths = zip(*tmp)
img_paths, mask_paths = list(img_paths), list(mask_paths)

# Split the data into train and test sets
train_imgs = img_paths[int(config.SPLIT_RATE * len(img_paths)):]
train_masks = mask_paths[int(config.SPLIT_RATE * len(mask_paths)):]
test_imgs = img_paths[:int(config.SPLIT_RATE * len(img_paths))]
test_masks = mask_paths[:int(config.SPLIT_RATE * len(mask_paths))]

# Save test image and mask paths
with open(config.TEST_IMGS_PATH, 'wb') as f:
    pickle.dump(test_imgs, f)
with open(config.TEST_MASKS_PATH, 'wb') as f:
    pickle.dump(test_masks, f)

# Create a transformation composition
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor()
])

# Load the train and test datasets
train_set = SegmentationDataset(train_imgs, train_masks, transform)
test_set = SegmentationDataset(test_imgs, test_masks, transform)
print('Train images: {}\n Test images: {}'.format(len(train_set), len(test_set)))

# Initiate the train and test loaders
train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, **config.KWARGS)
test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, **config.KWARGS)

# Create the model
model = UNet(config.ENC_CHANNELS, config.DEC_CHANNELS, config.N_CLASSES, config.IMAGE_SIZE).to(config.DEVICE)

# Initiate loss function and optimizer
loss_fn = CrossEntropyLoss()
optim = Adam(model.parameters(), lr=config.LEARNING_RATE)

# Calculate the number of train and test steps
train_steps = len(train_set) // config.BATCH_SIZE
test_steps = len(test_set) // config.BATCH_SIZE

# Initiate a train history
history = {'train_loss': [], 'test_loss': []}

# Initiate the timer
start_time = time.time()

# Train the model
for epoch in range(config.EPOCHS):
    print('EPOCH: {}'.format(epoch))
    # Set the model in learning mode
    model.train()
    # Initiate the train and test loss values
    train_loss = 0.
    test_loss = 0.

    # Loop through the training data
    for i, (images, targets) in enumerate(train_loader):
        # Move data to the device
        images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
        # Forward pass
        pred = model(images)
        loss = loss_fn(pred, targets)
        # Backward pass
        optim.zero_grad()
        loss.backward()
        # Update weights
        optim.step()
        # Update the train loss value
        train_loss += loss

    # Evaluate the model
    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()
        # Loop through the test data
        for (images, targets) in test_loader:
            # Move data to the device
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
            # Predict values and calculate the loss
            pred = model(images)
            loss = loss_fn(pred, targets)
            # Update the test loss value
            test_loss += loss

    # Calculate the average train and loss values
    avg_train_loss = train_loss / train_steps
    avg_test_loss = test_loss / test_steps
    # Update the training history
    history['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    history['test_loss'].append(avg_test_loss.cpu().detach().numpy())

    print('Train loss: {} Test loss: {}'.format(avg_train_loss, avg_test_loss))

# Display the training time
end_time = time.time() - start_time
print('Total time: {}s'.format(end_time))

# Save the model and history dict
torch.save(model, config.MODEL_PATH)
with open(config.HISTORY_PATH, 'wb', ) as f:
    pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
