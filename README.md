# Motion-enhanced Cardiac Anatomy Segmentation via an Insertable Temporal Attention Module

<p align="justify">
Cardiac anatomy segmentation is crucial for assessing cardiac morphology and function, aiding diagnosis and intervention. Deep learning (DL) improves accuracy over traditional methods, and recent studies show that adding motion information can enhance segmentation further. However, current methods either increase input dimensionality, making them computationally expensive, or use suboptimal techniques like non-DL registration, non-attention networks, or single-headed attention.
</p>

<p align="justify">
We propose a novel, computation-efficient approach using a <strong>scalable Temporal Attention Module (TAM)</strong> for motion enhancement and improved performance. TAM features a <strong>multi-headed, KQV projection cross-attention architecture</strong> and can be easily integrated into existing CNN- or Transformer-based networks, offering flexibility for future implementations.
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

Our results confirm that **TAM enhances motion-aware segmentation** while maintaining computational efficiency, making it a promising addition to future deep learning-based cardiac segmentation methods **(details will be in the paper)**.

# 📌 Result Synopsis
<table>
  <caption>Results of integrating our novel TAM with CNN- and Transformer-based segmentation models using the public <strong>CAMUS dataset</strong>. The improvements introduced by the TAM are highlighted in bold. The paper describes the <strong>PIA metric</strong>, which calculates the percentage of the total segmentation area accounted for by such “island areas,” defined as any segmentation mass that is not the largest and that is disconnected from the largest mass. <strong>PIA</strong> measures anatomical plausibility.</caption>
  <thead>
    <tr>
      <th rowspan="2"><strong>Methods</strong></th>
      <th colspan="4">Class-wise HD (mm) ($\downarrow$)</th>
      <th colspan="4">The average of the anatomical organs</th>
    </tr>
    <tr>
      <th><strong>LV<sub>MYO</sub></strong></th>
      <th><strong>LV<sub>ENDO</sub></strong></th>
      <th><strong>LV<sub>EPI</sub></strong></th>
      <th><strong>LA</strong></th>
      <th><strong>DSC($\uparrow$)</strong></th>
      <th><strong>HD(mm)($\downarrow$)</strong></th>
      <th><strong>MASD(mm)($\downarrow$)</strong></th>
      <th><strong><span style="font-weight: bold;">PIA</span>(%)($\downarrow$)</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>UNet</td>
      <td>5.65</td>
      <td>4.21</td>
      <td>5.67</td>
      <td>4.91</td>
      <td>0.913</td>
      <td>5.11</td>
      <td>1.13</td>
      <td>2.05</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-UNet</td>
      <td><strong>4.05</strong></td>
      <td><strong>3.07</strong></td>
      <td><strong>3.89</strong></td>
      <td><strong>3.52</strong></td>
      <td><span style="color: blue;"><strong>0.922</strong></span></td>
      <td><strong>3.63</strong></td>
      <td><span style="color: blue;"><strong>0.96</strong></span></td>
      <td><strong>0.68</strong></td>
    </tr>
    <tr>
      <td>FCN8s</td>
      <td>6.80</td>
      <td>5.44</td>
      <td>6.01</td>
      <td>7.26</td>
      <td>0.899</td>
      <td>6.38</td>
      <td>1.33</td>
      <td>0.58</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-FCN8s</td>
      <td><span style="color: blue;"><strong>3.60</strong></span></td>
      <td><span style="color: blue;"><strong>3.04</strong></span></td>
      <td><span style="color: blue;"><strong>3.33</strong></span></td>
      <td><span style="color: blue;"><strong>3.27</strong></span></td>
      <td><strong>0.921</strong></td>
      <td><span style="color: blue;"><strong>3.31</strong></span></td>
      <td><strong>0.98</strong></td>
      <td><span style="color: blue;"><strong>0.02</strong></span></td>
    </tr>
    <tr>
      <td>UNetR</td>
      <td>8.03</td>
      <td>5.59</td>
      <td>7.71</td>
      <td>8.35</td>
      <td>0.897</td>
      <td>7.42</td>
      <td>1.43</td>
      <td>2.43</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-UNetR</td>
      <td><strong>6.08</strong></td>
      <td><strong>4.62</strong></td>
      <td><strong>5.86</strong></td>
      <td><strong>6.05</strong></td>
      <td><strong>0.904</strong></td>
      <td><strong>5.65</strong></td>
      <td><strong>1.24</strong></td>
      <td><strong>0.92</strong></td>
    </tr>
    <tr>
      <td>SwinUNetR</td>
      <td>8.33</td>
      <td>5.60</td>
      <td>8.24</td>
      <td>6.41</td>
      <td>0.888</td>
      <td>7.15</td>
      <td>1.52</td>
      <td>2.67</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-SwinUNetR</td>
      <td><strong>5.63</strong></td>
      <td><strong>4.25</strong></td>
      <td><strong>5.32</strong></td>
      <td><strong>4.11</strong></td>
      <td><strong>0.913</strong></td>
      <td><strong>4.83</strong></td>
      <td><strong>1.15</strong></td>
      <td><strong>1.32</strong></td>
    </tr>
  </tbody>
</table>

<table>
  <caption>Results of integrating our novel 3D-TAM with 3D-CNN- and Transformer-based segmentation models using public <strong>3D echocardiography (MITEA)</strong>. Improvements introduced by the TAM are highlighted in bold. </caption>
  <thead>
    <tr>
      <th rowspan="2"><strong>Methods</strong></th>
      <th colspan="4">Class-wise HD (mm) ($\downarrow$)</th>
      <th colspan="4">The average of the anatomical organs</th>
    </tr>
    <tr>
      <th><strong>LV<sub>MYO</sub></strong></th>
      <th><strong>LV<sub>ENDO</sub></strong></th>
      <th><strong>LV<sub>EPI</sub></strong></th>
      <th><strong>LA</strong></th>
      <th><strong>DSC ($\uparrow$)</strong></th>
      <th><strong>HD (mm) ($\downarrow$)</strong></th>
      <th><strong>MASD (mm) ($\downarrow$)</strong></th>
      <th><strong><span style="font-weight: bold;">PIA</span> (%) ($\downarrow$)</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>UNet</td>
      <td>14.58</td>
      <td>9.89</td>
      <td>12.31</td>
      <td>-</td>
      <td>0.830</td>
      <td>12.26</td>
      <td>2.03</td>
      <td>0.30</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-UNet</td>
      <td><strong>11.47</strong></td>
      <td><strong>9.14</strong></td>
      <td><strong>10.84</strong></td>
      <td>-</td>
      <td><strong>0.833</strong></td>
      <td><strong>10.48</strong></td>
      <td><strong>1.97</strong></td>
      <td><strong>0.16</strong></td>
    </tr>
    <tr>
      <td>FCN8s</td>
      <td>12.07</td>
      <td>11.95</td>
      <td>10.95</td>
      <td>-</td>
      <td>0.828</td>
      <td>11.66</td>
      <td>2.06</td>
      <td>1.07</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-FCN8s</td>
      <td><span style="color: blue;"><strong>9.27</strong></span></td>
      <td><span style="color: blue;"><strong>7.59</strong></span></td>
      <td><span style="color: blue;"><strong>8.24</strong></span></td>
      <td>-</td>
      <td><span style="color: blue;"><strong>0.836</strong></span></td>
      <td><span style="color: blue;"><strong>8.37</strong></span></td>
      <td><span style="color: blue;"><strong>1.93</strong></span></td>
      <td><span style="color: blue;"><strong>0.22</strong></span></td>
    </tr>
    <tr>
      <td>UNetR</td>
      <td>13.39</td>
      <td>11.85</td>
      <td>12.74</td>
      <td>-</td>
      <td>0.806</td>
      <td>12.66</td>
      <td>2.34</td>
      <td>0.53</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-UNetR</td>
      <td><strong>10.70</strong></td>
      <td><strong>9.56</strong></td>
      <td><strong>9.96</strong></td>
      <td>-</td>
      <td><strong>0.814</strong></td>
      <td><strong>10.07</strong></td>
      <td><strong>2.21</strong></td>
      <td><strong>0.38</strong></td>
    </tr>
    <tr>
      <td>SwinUNetR</td>
      <td>10.95</td>
      <td>10.10</td>
      <td>10.25</td>
      <td>-</td>
      <td>0.818</td>
      <td>10.43</td>
      <td>2.27</td>
      <td>0.36</td>
    </tr>
    <tr>
      <td><strong>TAM</strong>-SwinUNetR</td>
      <td><strong>9.67</strong></td>
      <td><strong>8.67</strong></td>
      <td><strong>9.01</strong></td>
      <td>-</td>
      <td><strong>0.823</strong></td>
      <td><strong>9.12</strong></td>
      <td><strong>2.12</strong></td>
      <td><strong>0.23</strong></td>
    </tr>
  </tbody>
</table>

<table>
  <caption>Comparison of segmentation performance on the CAMUS dataset across state-of-the-art methods and our proposed motion-aware TAM-based segmentation models. Best-performing metrics are highlighted in bold.</caption>
  <thead>
    <tr>
      <th rowspan="2">Methods (motion?)</th>
      <th colspan="3">LV<sub>MYO</sub></th>
      <th colspan="3">LV<sub>ENDO</sub></th>
      <th colspan="3">LV<sub>EPI</sub></th>
      <th colspan="3">LA</th>
    </tr>
    <tr>
      <th>DSC (↑)</th>
      <th>HD (↓)</th>
      <th>MASD (↓)</th>
      <th>DSC (↑)</th>
      <th>HD (↓)</th>
      <th>MASD (↓)</th>
      <th>DSC (↑)</th>
      <th>HD (↓)</th>
      <th>MASD (↓)</th>
      <th>DSC (↑)</th>
      <th>HD (↓)</th>
      <th>MASD (↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a>(✘)</td>
      <td>0.864</td>
      <td>5.65</td>
      <td>1.10</td>
      <td>0.927</td>
      <td>4.21</td>
      <td>1.06</td>
      <td>0.954</td>
      <td>5.67</td>
      <td>1.15</td>
      <td>0.904</td>
      <td>4.91</td>
      <td>1.21</td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/abs/2201.01266" target="_blank">SwinUNetR</a>(✘)</td>
      <td>0.834</td>
      <td>8.33</td>
      <td>1.41</td>
      <td>0.908</td>
      <td>5.60</td>
      <td>1.42</td>
      <td>0.939</td>
      <td>8.24</td>
      <td>1.56</td>
      <td>0.869</td>
      <td>6.41</td>
      <td>1.68</td>
    </tr>
    <tr>
      <td><a href="https://ieeexplore.ieee.org/document/8051114" target="_blank">ACNN</a>(✘)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.918</td>
      <td>5.90</td>
      <td>1.80</td>
      <td>0.946</td>
      <td>6.35</td>
      <td>1.95</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://ieeexplore.ieee.org/document/10569083" target="_blank">BEASNet</a>(✘)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.915</td>
      <td>6.0</td>
      <td>1.95</td>
      <td>0.943</td>
      <td>6.35</td>
      <td>2.15</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.sciencedirect.com/science/article/pii/S0950705124010281?via%3Dihub" target="_blank">UB<sup>2</sup>DNet</a>(✘)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.858</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/abs/2308.13790" target="_blank">FFPN-R</a>(✘)</td>
      <td>0.850</td>
      <td>3.65</td>
      <td>-</td>
      <td>0.924</td>
      <td>3.05</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.888</td>
      <td>3.80</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.sciencedirect.com/science/article/pii/S1361841520302371" target="_blank">PLANet</a>(✘)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><strong style="color:blue;">0.944</strong></td>
      <td>4.14</td>
      <td>1.26</td>
      <td>0.957</td>
      <td>5.0</td>
      <td>1.72</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.sciencedirect.com/science/article/pii/S1746809424006918" target="_blank">CoSTUNet</a>(✘)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.916</td>
      <td>6.55</td>
      <td>-</td>
      <td>0.837</td>
      <td>7.65</td>
      <td>-</td>
      <td>0.875</td>
      <td>6.70</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.sciencedirect.com/science/article/pii/S136184152400166X" target="_blank">I<sup>2</sup>UNet</a>(✘)</td>
      <td>0.873</td>
      <td>4.72</td>
      <td>1.03</td>
      <td>0.933</td>
      <td>3.49</td>
      <td>1.02</td>
      <td>0.956</td>
      <td>4.39</td>
      <td>1.09</td>
      <td>0.910</td>
      <td>4.25</td>
      <td>1.19</td>
    </tr>
    <tr>
       <td>Our TAM-I<sup>2</sup>UNet(✔)</td>
      <td>0.872</td>
      <td>4.19</td>
      <td>1.03</td>
      <td>0.933</td>
      <td><strong style="color:blue;">3.02</strong></td>
      <td>0.972</td>
      <td>0.956</td>
      <td>3.92</td>
      <td>1.06</td>
      <td>0.913</td>
      <td>3.74</td>
      <td>1.11</td>
    </tr>
    <tr>
       <td>Our TAM-FCN8s(✔)</td>
      <td><strong>0.876</strong></td>
      <td><strong>3.60</strong></td>
      <td><strong>0.973</strong></td>
      <td>0.935</td>
      <td>3.04</td>
      <td><strong>0.949</strong></td>
      <td><strong>0.959</strong></td>
      <td><strong>3.33</strong></td>
      <td><strong>0.961</strong></td>
      <td><strong>0.916</strong></td>
      <td><strong>3.27</strong></td>
      <td><strong>1.06</strong></td>
    </tr>
    <tr>
      <td><a href="https://ieeexplore.ieee.org/document/9946374" target="_blank">SOCOF</a>(✔)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.932</td>
      <td>3.21</td>
      <td>1.40</td>
      <td>0.953</td>
      <td>4.0</td>
      <td>1.65</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a href="https://link.springer.com/chapter/10.1007/978-3-030-59713-9_60" target="_blank">CLAS</a>(✔)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.935</td>
      <td>4.60</td>
      <td>1.40</td>
      <td>0.958</td>
      <td>4.85</td>
      <td>1.55</td>
      <td>0.915</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>


# 📌 Implementation

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

