# Importing necessary libraries
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import config  # Assuming config.py is present in the same directory
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from monai.losses import *
from monai.metrics import *
from monai.utils import set_determinism, first

# Function to convert integer labels to one-hot encoding
def make_one_hot(labels, device, C=2):
    '''
    Converts integer labels to one-hot encoding for semantic segmentation tasks.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        Shape: N x 1 x H x W, where N is the batch size. 
        Each value is an integer representing the correct class label.
    device : torch.device
        The device (CPU or GPU) where the output tensor should be allocated.
    C : int, optional, default=2
        The number of classes in the segmentation task.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        Shape: N x C x H x W, where C is the number of classes. This is the one-hot encoded tensor.
    '''
    # Ensure labels are of type LongTensor
    labels = labels.long()
    
    # Move labels to the specified device (CPU or GPU)
    labels = labels.to(device)
    
    # Create a zero-initialized one-hot tensor with the appropriate dimensions
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_().to(device)
    
    # Use scatter_ to set the corresponding class index to 1 for each pixel
    target = one_hot.scatter_(1, labels.data, 1)
    
    # Convert the result to a torch.autograd.Variable
    target = Variable(target)
        
    return target


# Define the Dice2DMultiClass class, which inherits from the torch.nn.Module class
class Dice2DMultiClass(nn.Module):
    # Constructor method that initializes the class
    def __init__(self, num_classes, smooth=1e-5):
        '''
        Initializes the Dice2DMultiClass loss function.

        Parameters
        ----------
        num_classes : int
            The number of classes in the segmentation task.
        smooth : float, optional, default=1e-5
            A small constant added to avoid division by zero during Dice coefficient calculation.
        '''
        # Call the constructor of the parent class (torch.nn.Module)
        super(Dice2DMultiClass, self).__init__()
        
        # Set the number of classes and the smoothing factor as attributes
        self.num_classes = num_classes
        self.smooth = smooth

    # Forward method to compute the Dice loss during the forward pass
    def forward(self, prediction, target):
        '''
        Computes the Dice loss for multi-class segmentation.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted output from the model, with shape (N, C, H, W), where
            N is the batch size, C is the number of classes, and H, W are the spatial dimensions.
        target : torch.Tensor
            The ground truth target, with shape (N, C, H, W), where each pixel's value corresponds
            to the class label for the respective pixel.

        Returns
        -------
        dice_losses : list
            A list containing the Dice coefficient for each class in the segmentation task.
        '''
        # Convert the target to one-hot encoding (Note: This step may not be necessary depending on how the target is provided)
        target_one_hot = target

        # Initialize an empty list to store individual Dice losses for each class
        dice_losses = []

        # Iterate over each class
        for class_idx in range(self.num_classes):
            # Extract the predicted probabilities and corresponding one-hot encoded target for the current class
            class_prediction = prediction[:, class_idx, :, :]
            class_target = target_one_hot[:, class_idx, :, :]

            # Compute the intersection, union, and Dice coefficient for the current class
            intersection = torch.sum(class_prediction * class_target)
            union = torch.sum(class_prediction) + torch.sum(class_target)
            dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)

            # Set the Dice loss for the current class
            dice_loss = dice_coefficient

            # Append the Dice loss to the list
            dice_losses.append(dice_loss)

        # Calculate the average Dice loss across all classes and return it
        return dice_losses



# Define a custom DiceLoss for 2D multi-class segmentation
class DiceLoss2DMultiClass(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        '''
        Initializes the DiceLoss2DMultiClass loss function.

        Parameters
        ----------
        num_classes : int
            The number of classes in the segmentation task.
        smooth : float, optional, default=1e-5
            A small constant added to avoid division by zero during Dice coefficient calculation.
        '''
        super(DiceLoss2DMultiClass, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, prediction, target):
        '''
        Computes the Dice loss for multi-class segmentation.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted output from the model, with shape (N, C, H, W), where
            N is the batch size, C is the number of classes, and H, W are the spatial dimensions.
        target : torch.Tensor
            The ground truth target, with shape (N, C, H, W), already one-hot encoded.

        Returns
        -------
        dice_loss : torch.Tensor
            The mean Dice loss across all classes in the segmentation task.
        '''
        # Convert the target to one-hot encoding (Note: target is already one-hot encoded)
        target_one_hot = target

        # List to store Dice loss for each class
        dice_losses = []

        # Iterate over each class (starting from 1 to exclude background class)
        for class_idx in range(1, self.num_classes):
            # Extract the predicted probabilities and corresponding one-hot encoded target for the current class
            class_prediction = prediction[:, class_idx, :, :]
            class_target = target_one_hot[:, class_idx, :, :]

            # Calculate Dice coefficient and Dice loss for the current class
            intersection = torch.sum(class_prediction * class_target)
            union = torch.sum(class_prediction) + torch.sum(class_target)
            dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1. - dice_coefficient

            # Append the Dice loss for the current class to the list
            dice_losses.append(dice_loss)

        # Average the Dice losses across all classes and return the result
        return torch.mean(torch.stack(dice_losses))



# Function for thresholding the prediction mask
def thresholded(predMask, LB, UB):
    '''
    Applies thresholding to the predicted mask, categorizing it into three classes:
    - Background class for values below the lower bound (LB)
    - Myocardium class for values between the lower and upper bounds (LB, UB)
    - Left ventricle class for values above the upper bound (UB)

    Parameters
    ----------
    predMask : torch.Tensor
        The predicted mask with shape (N, C, H, W), where N is the batch size, C is the number of classes,
        and H, W are the spatial dimensions of the mask.
    LB : float
        The lower bound threshold for classifying the mask into Myocardium or Background class.
    UB : float
        The upper bound threshold for classifying the mask into Myocardium or Left Ventricle class.

    Returns
    -------
    thresholded_predMask : torch.Tensor
        The thresholded mask with values assigned to the corresponding classes:
        - Background (0)
        - Myocardium (1)
        - Left Ventricle (2)
    '''
    # Thresholding the prediction by cloning the original prediction mask
    thresholded_predMask = predMask.clone()

    # Define the classes based on thresholding
    background_class = torch.zeros_like(predMask)  # Values below LB are assigned to background
    myo_class = torch.ones_like(predMask)          # Values between LB and UB are assigned to myocardium
    lv_class = 2 * torch.ones_like(predMask)       # Values above UB are assigned to left ventricle

    # Apply the lower bound threshold to classify background
    thresholded_predMask_ = torch.where(thresholded_predMask < LB, background_class, thresholded_predMask)
    
    # Apply the upper bound threshold to classify left ventricle
    thresholded_predMask_ = torch.where(thresholded_predMask_ > UB, lv_class, thresholded_predMask_)
    
    # Apply the range between LB and UB to classify myocardium
    thresholded_predMask_ = torch.where(
        (thresholded_predMask_ <= UB) & (thresholded_predMask_ >= LB), 
        myo_class, thresholded_predMask_
    )

    return thresholded_predMask_




def plot_outputs(img, mask, img_size):
    '''
    Plots the overlay of predicted masks on the corresponding input images.

    Parameters
    ----------
    img : torch.Tensor
        The input image tensor with shape (N, C, H, W), where N is the batch size,
        C is the number of channels, and H, W are the spatial dimensions of the image.
    mask : torch.Tensor
        The predicted mask tensor with shape (N, 1, H, W), where N is the batch size,
        and H, W are the spatial dimensions of the mask.
    img_size : int
        The size (height and width) to which the image and mask should be reshaped 
        for visualization purposes.

    Returns
    -------
    None
        The function displays the overlay of the input image and mask using matplotlib.
    '''
    # Create a figure with a size of 15x3 inches
    plt.figure(figsize=(15, 3))

    # Loop over the first 5 images and their corresponding masks
    for i in range(5):
        # Reshape and detach the mask and image tensors, then convert them to NumPy arrays
        rec_mask_ = mask[i, 0, :, :].reshape(img_size, img_size).detach().cpu().numpy() 
        rec_img_ = img[i, 0, :, :].reshape(img_size, img_size).detach().cpu().numpy()  

        # Convert the mask to uint8 type and scale the image values to 0-255
        rec_mask_ = rec_mask_.astype(np.uint8)
        rec_img_ = (255 * rec_img_).astype(np.uint8)

        # Overlay the myocardium (MYO) mask on the input image
        new = overlayMYO(rec_img_, rec_mask_, 0.4)   

        # Create a subplot and display the overlayed image
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(new, cmap='gist_gray')  

        # Hide the x and y axes for better visualization
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
           
    # Show the plot
    plt.show()



def overlayMYO(originalImage, maskImage, alpha):
    """
    Overlays the borders of a mask on an image with specified transparency.

    Parameters:
    originalImage (np.ndarray): The original image.
    maskImage (np.ndarray): The mask image.
    alpha (float): Transparency factor for the overlay (0 to 1).

    Returns:
    np.ndarray: The resulting image with the overlay.
    """

    # Step 1: Load the image and mask
    if isinstance(originalImage, str):
        originalImage = cv2.imread(originalImage)
    if isinstance(maskImage, str):
        maskImage = cv2.imread(maskImage, cv2.IMREAD_GRAYSCALE)

    originalImage = np.repeat(originalImage[:, :, np.newaxis], 3, axis=2)
    
    # Step 2: Create a border overlay for label 100
    # Detect edges of the region with label 100
    edges = cv2.Canny((maskImage == 1).astype(np.uint8) * 255, 100, 200)

    # Create a color version of the mask for borders
    borderOverlay = np.zeros_like(originalImage)

    # Assign green color to the edges (you can choose any color)
    borderOverlay[edges != 0] = [0, 255, 0]

    # Step 3: Overlay the mask onto the image
    # Convert the original image to double for blending
    originalImage = originalImage.astype(np.float32) / 255.0

    # Convert borderOverlay to double
    borderOverlay = borderOverlay.astype(np.float32) / 255.0

    # Blend the images
    overlaidImage = (1 - alpha) * originalImage + alpha * borderOverlay

    # Convert the result back to uint8
    overlaidImage = (overlaidImage * 255).astype(np.uint8)

    # Return the resulting image
    return overlaidImage


# Define CrossEntropyLoss instead of MSELoss
cross_entropy_loss = nn.CrossEntropyLoss()

# Initialize Dice metric and loss functions
dice_multiclass_metric = Dice2DMultiClass(num_classes=config.num_classes)
dice_multiclass_metric_EPI = Dice2DMultiClass(num_classes=2)
dice_multiclass_Loss = DiceLoss2DMultiClass(num_classes=config.num_classes)

def train_helper(pred, actual, device):
    '''
    Helper function to calculate Dice loss, cross-entropy loss, and Dice scores for multi-class segmentation.

    Parameters
    ----------
    pred : torch.Tensor
        The predicted segmentation output from the model, with shape (N, C, H, W),
        where N is the batch size, C is the number of classes, and H, W are the spatial dimensions.
    actual : torch.Tensor
        The ground truth segmentation mask, with shape (N, 1, H, W).
    device : torch.device
        The device (CPU or GPU) where the tensors are located.

    Returns
    -------
    DSC_LOSS : torch.Tensor
        The Dice loss between the predicted and actual segmentation masks.
    CE_LOSS : torch.Tensor
        The cross-entropy loss between the predicted and actual segmentation masks.
    DSC_MYO_ENDO : torch.Tensor
        The Dice score for Myocardium and Endocardium classes.
    DSC_EPI : torch.Tensor
        The Dice score for Epicardium class.
    '''
    # Pred segmentation
    segmented_pred = torch.argmax(pred, dim=1).unsqueeze(1).to(torch.float32)
    segmented_pred = segmented_pred.clone().detach().requires_grad_(True)

    # Calculate Dice loss
    DSC_LOSS = dice_multiclass_Loss(make_one_hot(actual, device, C=config.num_classes),
                                    make_one_hot(segmented_pred, device, C=config.num_classes)).requires_grad_(True)

    # Cross-entropy loss
    CE_LOSS = cross_entropy_loss(pred, actual.squeeze(1).long())

    # Calculate Dice score for Myocardium and Endocardium
    DSC_MYO_ENDO = dice_multiclass_metric(make_one_hot(actual, device, C=config.num_classes),
                                          make_one_hot(segmented_pred, device, C=config.num_classes))

    # Create mask for Epicardium class
    temp_train_msk = actual.clone()
    temp_train_msk[temp_train_msk == 0] = 0
    temp_train_msk[temp_train_msk == 1] = 1
    temp_train_msk[temp_train_msk == 2] = 1
    temp_train_msk[temp_train_msk == 3] = 0
    
    temp_segmented_mask_train = segmented_pred.clone()
    temp_segmented_mask_train[temp_segmented_mask_train == 0] = 0
    temp_segmented_mask_train[temp_segmented_mask_train == 1] = 1
    temp_segmented_mask_train[temp_segmented_mask_train == 2] = 1
    temp_segmented_mask_train[temp_segmented_mask_train == 3] = 0

    # Calculate Dice score for Epicardium
    DSC_EPI = dice_multiclass_metric_EPI(make_one_hot(temp_train_msk, device, C=2),
                                         make_one_hot(temp_segmented_mask_train, device, C=2))

    return DSC_LOSS, CE_LOSS, DSC_MYO_ENDO, DSC_EPI


# Define FocalLoss class (if not already defined)
class FocalLoss(nn.Module):
    '''
    Focal Loss function, designed to address the class imbalance problem in segmentation tasks.

    Parameters
    ----------
    alpha : float, optional, default=1
        A weighting factor to balance the importance of positive vs negative examples.
    gamma : float, optional, default=2
        A focusing parameter that adjusts the rate at which easy examples are down-weighted.
    reduction : str, optional, default='mean'
        Specifies the reduction to apply to the output: 'mean' or 'sum'.
    '''
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        '''
        Forward pass for calculating the Focal Loss.

        Parameters
        ----------
        inputs : torch.Tensor
            The predicted logits (raw output) from the model, with shape (N, C, H, W).
        targets : torch.Tensor
            The ground truth labels, with shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            The calculated Focal Loss.
        '''
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability of correct classification
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply the specified reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Focal loss instead of cross-entropy
FL = FocalLoss(alpha=1, gamma=2)

# Helper function for training with Focal Loss
def train_helper_FL(pred, actual, device):
    '''
    Helper function to calculate Dice loss, Focal loss, and Dice scores for multi-class segmentation
    with Focal Loss used instead of cross-entropy.

    Parameters
    ----------
    pred : torch.Tensor
        The predicted segmentation output from the model, with shape (N, C, H, W).
    actual : torch.Tensor
        The ground truth segmentation mask, with shape (N, 1, H, W).
    device : torch.device
        The device (CPU or GPU) where the tensors are located.

    Returns
    -------
    DSC_LOSS : torch.Tensor
        The Dice loss between the predicted and actual segmentation masks.
    FOCAL_LOSS : torch.Tensor
        The Focal loss between the predicted and actual segmentation masks.
    DSC_MYO_ENDO : torch.Tensor
        The Dice score for Myocardium and Endocardium classes.
    DSC_EPI : torch.Tensor
        The Dice score for Epicardium class.
    '''
    # Pred segmentation
    segmented_pred = torch.argmax(pred, dim=1).unsqueeze(1).to(torch.float32)
    segmented_pred = segmented_pred.clone().detach().requires_grad_(True)

    # Calculate Dice loss
    DSC_LOSS = dice_multiclass_Loss(make_one_hot(actual, device, C=config.num_classes),
                                    make_one_hot(segmented_pred, device, C=config.num_classes)).requires_grad_(True)

    # Calculate Focal Loss
    FOCAL_LOSS = FL(pred, make_one_hot(actual, device, C=config.num_classes))

    # Calculate Dice score for Myocardium and Endocardium
    DSC_MYO_ENDO = dice_multiclass_metric(make_one_hot(actual, device, C=config.num_classes),
                                          make_one_hot(segmented_pred, device, C=config.num_classes))

    # Create mask for Epicardium class
    temp_train_msk = actual.clone()
    temp_train_msk[temp_train_msk == 0] = 0
    temp_train_msk[temp_train_msk == 1] = 1
    temp_train_msk[temp_train_msk == 2] = 1
    temp_train_msk[temp_train_msk == 3] = 0
    
    temp_segmented_mask_train = segmented_pred.clone()
    temp_segmented_mask_train[temp_segmented_mask_train == 0] = 0
    temp_segmented_mask_train[temp_segmented_mask_train == 1] = 1
    temp_segmented_mask_train[temp_segmented_mask_train == 2] = 1
    temp_segmented_mask_train[temp_segmented_mask_train == 3] = 0

    # Calculate Dice score for Epicardium
    DSC_EPI = dice_multiclass_metric_EPI(make_one_hot(temp_train_msk, device, C=2),
                                         make_one_hot(temp_segmented_mask_train, device, C=2))

    return DSC_LOSS, FOCAL_LOSS, DSC_MYO_ENDO, DSC_EPI



def visualize_sample(imgs, msks, img_size, title_prefixes):
    """
    Visualize the images and masks samples for multiple targets.

    Parameters:
    - imgs: List of image tensors to visualize.
    - msks: List of mask tensors to visualize.
    - img_size: The size to reshape the image and mask for visualization.
    - title_prefixes: List of prefixes for the titles of the subplots, one per target.
    """
    num_targets = len(title_prefixes)
    plt.figure(figsize=(15, 5 * num_targets))

    for i in range(num_targets):
        # Display image
        plt.subplot(num_targets, 5, 5 * i + 1)
        plt.title(f"{title_prefixes[i]} Image")
        plt.imshow(imgs[i].reshape(img_size, img_size), cmap="gray")
        plt.axis('off')

        # Display mask channels
        for j, label in enumerate(['BACK', 'MYO', 'LV', 'LA']):
            plt.subplot(num_targets, 5, 5 * i + 2 + j)
            plt.title(f"{title_prefixes[i]} Mask {label}")
            plt.imshow(msks[i][j].reshape(img_size, img_size), cmap="gray")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_dataset_sample(dataset, targets, device='cpu', num_classes=3, img_size=128, dataset_type='Train'):
    """
    Process and visualize samples for multiple targets from the specified dataset.

    Parameters:
    - dataset: The dataset to visualize (either trainData or valData).
    - targets: A list of target keys to extract images and masks.
    - device: The device to use for tensor operations (e.g., 'cpu').
    - num_classes: The number of classes for one-hot encoding.
    - img_size: The size to reshape the images and masks for visualization.
    - dataset_type: A string to identify the dataset type ('Train' or 'Validation').
    """
    imgs = []
    msks = []
    title_prefixes = []
    temp = first(dataset)  # Assuming first function extracts the first sample from dataset

    for target in targets:
        # Extract and preprocess data
        img = temp[target]["image"]
        msk = temp[target]["mask"]
        msk = make_one_hot(msk, device, num_classes)
        
        # Append to lists
        imgs.append(img[0])
        msks.append(msk[0])
        title_prefixes.append(f"{dataset_type} {target}")

    # Visualize the samples
    visualize_sample(imgs, msks, img_size, title_prefixes)