"""
Configuration settings for training the models.
This script defines hyperparameters, dataset properties, and training conditions.
"""

# Flag to enable/disable training on multiple GPUs
# Set to True if using multiple GPUs, otherwise False for a single GPU
double_GPU = False

# Flags to enable/disable shuffling of training and validation datasets
shuffle_ = True  # Shuffle training data to improve generalization
shuffle_val = True  # Shuffle validation data to prevent order bias

# Batch sizes for different dataset splits
trainBatch = 8  # Number of training samples per batch
valBatch = 8  # Number of validation samples per batch
testBatch = 8  # Number of test samples per batch

# Number of training epochs
Epochs = 3  # Total number of times the model will see the entire dataset

# Learning rate for optimizer
LR = 1e-4  # Step size for updating model parameters

# Image size (assumed to be square: height = width)
img_size = 256  # Dimensions of input images

# Number of worker threads for data loading
num_workers = 0  # 0 means data loading happens in the main process

# Range for a specific parameter (potentially a threshold for a metric)
lower_bound = 0.999  # Lower limit of the range
upper_bound = 1.99  # Upper limit of the range

# Number of output classes (segmentation classes, classification categories, etc.)
num_classes = 4  # Total number of distinct labels in the dataset

# Validation interval (how often validation is performed)
val_interval = 1  # Perform validation after every epoch
