# Motion-enhanced Cardiac Anatomy Segmentation via an Insertable Temporal Attention Module

<p align="justify">
Cardiac anatomy segmentation is very useful for clinical assessment of cardiac morphology and function to inform diagnosis and intervention. Compared to traditional segmentation approaches, deep learning (DL) can improve accuracy significantly, and more recently, studies have shown that enhancing DL segmentation with motion information can further improve it. 
</p>

<p align="justify">
However, recent methods for injecting motion information either increase input dimensionality, which is computationally expensive, or use suboptimal approaches, such as non-DL registration, non-attention networks, or single-headed attention. 
</p>

<p align="justify">
Here, we present a novel, computation-efficient alternative where a <strong>scalable Temporal Attention Module (TAM)</strong> can be inserted into existing networks for motion enhancement and improved performance. TAM has a <strong>multi-headed, KQV projection cross-attention architecture</strong> and can be seamlessly integrated into a wide range of existing CNN- or Transformer-based networks, making it flexible for future implementations.
</p>

## Key Contributions:
- **Novel Temporal Attention Mechanism for Segmentation:**  
  - We present a new **Temporal Attention Module (TAM)**, a **multi-headed, temporal cross-time attention mechanism** based on **KQV projection**, that enables the network to effectively capture dynamic changes across temporal frames for motion-enhanced cardiac anatomy segmentation.

- **Flexible Integration into a Range of Segmentation Networks:**  
  - TAM can be **plug-and-play integrated** into a variety of established backbone segmentation architectures, including [UNet](https://arxiv.org/abs/1505.04597), [FCN8s](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf), [UNetR](https://arxiv.org/abs/2103.10504), [SwinUNetR](https://arxiv.org/abs/2201.01266), [I²UNet](https://www.sciencedirect.com/science/article/pii/S136184152400166X), [DT-VNet](https://ieeexplore.ieee.org/abstract/document/10752102), and others, arming them with motion-awareness. This provides a **simple and elegant** approach to implementing motion awareness in future networks.

- **Consistent Performance Across Multiple Settings:**  
  - **Generalizable** across different image types and qualities, from **2D to 3D cardiac datasets**.  
  - **Highly adaptable**, improving segmentation performance across various backbone architectures.  
  - **Computationally efficient**, adding minimal overhead and outperforming methods that increase input dimensionality.
 
- **Extensive evaluation on diverse cardiac datasets:**
  - **2D echocardiography ([CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/))**
  - **3D echocardiography ([MITEA](https://www.cardiacatlas.org/mitea/))**
  - **3D cardiac MRI ([ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/))**

Our results confirm that **TAM enhances motion-aware segmentation** while maintaining computational efficiency, making it a promising addition to future deep learning-based cardiac segmentation methods.

# 📌 Temporal Attention Module (TAM) - Dataset Preparation

This document provides an overview of how to **load, preprocess, and structure** cardiac imaging datasets (NIfTI format) for training **motion-aware segmentation networks**.

## 🔹 Overview
Cardiac image sequences typically consist of:
- **End-Diastolic (ED) frame**
- **End-Systolic (ES) frame**
- **Mid-systolic frames (optional intermediate frames between ED and ES)**

Our dataset preparation pipeline ensures:
✅ **Efficient loading of NIfTI images**  
✅ **Rescaling & Normalization** to a consistent resolution  
✅ **Preserving segmentation labels** during resizing  
✅ **Multi-frame integration** for temporal attention  

---

## 🛠️ **Pseudocode: Loading & Preprocessing**

### 1️⃣ **Read a NIfTI Image**
```python
function read_nifti_img(filepath, target_shape):
    # Load NIfTI file
    # Resize if needed
    # Normalize intensities
    # Return image tensor

### 2️⃣ **Read a NIfTI Mask**
```python
function read_nifti_mask(filepath, target_shape):
    # Load NIfTI mask file
    # Resize if needed (using nearest-neighbor interpolation)
    # Return mask tensor (long dtype for class labels)

class ReadDataset_TA(Dataset):
    function __init__(self, image_paths, mask_paths, num_mid_frames=None, transform=None):
        # Initialize dataset with paths and optional transformations

    function __getitem__(self, idx):
        # Load ED and ES frames and optionally Mid frames
        # Apply transformations or convert to tensor if no transformation is provided
        # Combine and return ED, ES, and Mid frames as a dictionary

