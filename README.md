# U-Net network for Semantic Segmentation

Implementation of the [U-Net model](https://arxiv.org/abs/1505.04597) using PyTorch framework for a multi-class semantic
segmentation on The Oxford-IIIT Pet Dataset

## Results

Columns: Image, Ground truth mask, Predicted segmentation mask

![Image](https://github.com/Kojec1/Oxford-IIIT-Pets-Segmentation/blob/main/output/pred.png)

### Training

![History](https://github.com/Kojec1/Oxford-IIIT-Pets-Segmentation/blob/main/output/history.png)

## Dataset

The dataset consists of 37 categories of dogs and cats with roughly 200 images for each class. Each image has an
associated ground truth mask. </br>
The dataset can be found here - https://www.robots.ox.ac.uk/~vgg/data/pets/