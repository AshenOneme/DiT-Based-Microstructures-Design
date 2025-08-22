<div align=center>
  
# Latent Diffusion–Driven Inverse Design of Damping Microstructures with Multiaxial Nonlinear Mechanical Targets

</div> 

> **_The dataset used in this study is extremely large; even after a 90% reduction, the numerical model still required 1.13 TB of memory. Public datasets for microstructure inverse design are currently very scarce. We are therefore pleased to share such datasets, and part of the curated data has already been made available at https://doi.org/10.1016/j.tws.2025.113865 and https://doi.org/10.1016/j.tws.2025.113865._**      

> **_The advantage of this pipeline lies in its integration of a complete workflow, covering automatic design, three-dimensional modeling, industrial manufacturing, and mechanical simulation._**


<!-- 逆向设计 -->
* ## 🧭 **_Overview of the workflow_**
The framework seamlessly integrates feature compression via a variational autoencoder-based TopoFormer, representative selection through latent-space clustering, and conditional latent diffusion transformers guided by full nonlinear shear and compression mechanical performance curves.
<div align=center>
  <img width="1000" src="Figs/1.png"/>
   <div align=center><strong>Fig. 1. Workflow of the generative framework for damping microstructures</strong></div>
</div><br>    

* ## ⚛️ **_Checkpoints_**
The dataset used in this study contains more than **_140,000 samples_**. By applying clustering and latent space transformation, the training complexity was reduced by **_over 90%_**. The checkpoints of the diffusion models are provided in the link below.     
[**_🔗The checkpoints of the Diffusion Models_**](https://github.com/AshenOneme/DiT-Based-Microstructures-Design/releases/tag/Checkpoint_Microstructure_Design)

* ## 🚀 **_TXT2CAE_**
The **_TXT2CAE_** tool can convert images into three-dimensional models, which can then be directly meshed in ABAQUS and further utilized for industrial manufacturing.[**_🔗TXT2CAE_**](https://github.com/AshenOneme/MultiDampGen)   
  <div align=center>
  <img width="1000" src="Figs/T2C.png"/>
   <div align=center><strong>Fig. 2. The TXT2CAE tool</strong></div>
</div><br>    

<!-- 对比 -->
* ## 🏦 **_Model complexity and quantitative evaluation metrics_**
The computational complexity and structural quality metrics of CondUNet-S, B, L, X and DiT are quantitatively compared in the following Table.
| Model | Params (M) | FLOPs (G) | MACs (G) | FID | COV | PREC | SSIM |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| CondUNet-S | 24.33 | 2.91 | 1.46 | 13.535 | 0.869 | 0.902 | 0.651 |
| CondUNet-B | 39.75 | 4.31 | 2.15 | 12.380 | 0.862	| 0.921	| 0.639 |
| CondUNet-L | 84.67 | 7.06	| 3.53 | 12.373	| 0.850	| 0.934	| 0.658 |
| CondUNet-X | 99.91 | 4.43 | 2.21 | 11.592	| 0.886	| 0.853	| 0.658 |
| **_Diffusion Transformer_** | **138.09** | **5.49**	| **2.75** | **11.367 ↓** | **0.889 ↑** | **0.943 ↑** | **0.676 ↑** |

<!-- 验证 -->
* ## 🔬 **_Results and validation_**
Experimental tests on selected inverse-designed microstructures showed good agreement with the finite element results and satisfied the predefined mechanical performance requirements.
<div align=center>
    <img width="1000" src="Figs/2.png"/>
   <div align=center><strong>Fig. 3. Generated results of different models for the same target</strong></div>
  <img width="1000" src="Figs/3.png"/>
   <div align=center><strong>Fig. 4. Validation of generated results</strong></div>
</div><br>

* ## 🛠️ **_Experimental process_**
Extensive experimental validations were conducted on the microstructures designed using the diffusion model. Both the proposed deep learning model and the finite element numerical model accurately reflect the real conditions.
<div align=center>
    <img width="1000" src="Figs/4.gif"/>
   <div align=center><strong>Fig. 5. Low-cycle reciprocating loading process</strong></div>
</div><br>    
