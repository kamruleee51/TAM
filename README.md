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
  We present a new **Temporal Attention Module (TAM)**, a **multi-headed, temporal cross-time attention mechanism** based on **KQV projection**, that enables the network to effectively capture dynamic changes across temporal frames for motion-enhanced cardiac anatomy segmentation.

- **Flexible Integration into a Range of Segmentation Networks:**  
  TAM can be **plug-and-play integrated** into a variety of established backbone segmentation architectures, including **UNet, FCN8s, UNetR, SwinUNetR, and I²UNet**, arming them with motion-awareness. This provides a **simple and elegant** approach to implementing motion awareness in future networks.

- **Consistent Performance Across Multiple Settings:**  
  - **Generalizable** across different image types and qualities, from **2D to 3D cardiac datasets**.  
  - **Highly adaptable**, improving segmentation performance across various backbone architectures.  
  - **Computationally efficient**, adding minimal overhead and outperforming methods that increase input dimensionality.
 
- **Extensive evaluation on diverse cardiac datasets:**
  - **2D echocardiography (CAMUS)**
  - **3D echocardiography (MITEA)**
  - **3D MRI (ACDC)**

Our results confirm that **TAM enhances motion-aware segmentation** while maintaining computational efficiency, making it a promising addition to future deep learning-based cardiac segmentation methods.

