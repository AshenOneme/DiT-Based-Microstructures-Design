<div align=center>
  
# Latent Diffusion–Driven Inverse Design of Damping Microstructures with Multiaxial Nonlinear Mechanical Targets
  
</div> 

* ## ⚛️ **_Datasets & Pre-trained models_**
  The multiscale microstructure dataset encompasses a total of *__50,000 samples__*. The dataset utilized in this study, along with the pre-trained weights of **_MultiDampGen_**, can be accessed through the link provided below.      
[**🔗The damping microstructure dataset**](https://github.com/AshenOneme/MultiDampGen)   

<!-- 逆向设计 -->
* ## 🧭 **_Overview of the workflow_**
The framework seamlessly integrates feature compression via a variational autoencoder-based TopoFormer, representative selection through latent-space clustering, and conditional latent diffusion transformers guided by full nonlinear shear and compression mechanical performance curves.
<div align=center>
  <img width="1000" src="Figs/1.png"/>
   <div align=center><strong>Fig. 1. Workflow of the generative framework for damping microstructures</strong></div>
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
| **Diffusion Transformer** | **138.09** | **5.49**	| **2.75** | **11.367 ↓** | **0.889 ↑** | **0.943 ↑** | **0.676 ↑** |

<!-- 验证 -->
* ## 🔬 **_Results and validation_**
Experimental tests on selected inverse-designed microstructures showed good agreement with the finite element results and satisfied the predefined mechanical performance requirements.
<div align=center>
    <img width="1000" src="Figs/2.png"/>
   <div align=center><strong>Fig. 2. Generated results of different models for the same target</strong></div>
  <img width="1000" src="Figs/3.png"/>
   <div align=center><strong>Fig. 3. Validation of generated results</strong></div>
    <img width="1000" src="Figs/4.gif"/>
   <div align=center><strong>Fig. 4. Low-cycle reciprocating loading process</strong></div>
</div><br>    
