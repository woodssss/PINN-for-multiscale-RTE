# Introduction
This project provides codes for paper [Solving multiscale steady radiative transfer equation using neural networks with uniform stability](https://link.springer.com/article/10.1007/s40687-022-00345-z). The main difficulty comes from
the significantly varying magnitude of the Knusen number. To overcome the multiscale issue, we propsed an novel PINN loss based on the micro macro decomposition for the steady state of radiative transfer equation(RTE). 


# Usage of code
The following codes require some input parameters. The epsi is the Knudsen number, bc_loss_weight is the weight of boundary loss
function, Nx is number of grid points on x direction, nl is number of hidden layers, nr is number of neurons in each hidden layer.
## 1D isotropic case with constant inflow B.C.
```
python rte_rg.py epsi bc_loss_weight Nx nl nr
```
## 1D isotropic case with boundary layer.
Firstly, compute the boundary layer corrector
```
python hsp_rb_msi.py
```
Then 
```
python rte_rg_hsp.py epsi bc_loss_weight Nx nl nr
```
## 1D anisotropic case with constant boundary
```
python rte_rg_aniso.py epsi bc_loss_weight Nx nl nr
```
## 1D anisotropic case with boundary layer
Firstly, compute the boundary layer corrector
```
python hsp_rb_at_msi.py
```
Then 
```
python rte_rg_hsp_aniso.py epsi bc_loss_weight Nx nl nr
```
## 1D two materials.
```
python rte_rg_sigma.py epsi bc_loss_weight Nx nl nr
```
## 1D scattering function
```
python rte_rg_epsix.py
```
## 1D Nonlinear RTE with constant boundary
```
python rte_rg_nonl.py epsi bc_loss_weight Nx nl nr
```
## 1D Nonlinear RTE with boundary layer
First, compute boundary layer corrector
```
python hsp_nonl_msi.py
```
Then 
```
python rte_rg_nonl_hsp.py epsi bc_loss_weight Nx nl nr
```
## 2D RTE with constant boundary
```
python rte_2D.py epsi bc_loss_weight Nx nl nr
```
## 2D RTE with square boundary involved boundary layer
Compute auxiliary hsp for each grid points on boundary parallelly
```
python hsp_rb_aux_para.py
```
Train 2D boundary layer corrector
```
python hsp_2d_aux_limit.py
```
Then
```
python rte_2d_hsp.py epsi bc_loss_weight Nx nl nr
```
# Figures in paper
hsp_1d
```
python hsp_rb_load.py
```
hsp_2d_H and hsp_2d_reflec
```
python load_Gamma_aux.py
```
rg_1d_epsi1, rg_1d_epsizpzz1 and hsp_1d_bl_epsi1
```
python load_rte.py
```
hsp_1d_bl_epsizppz1
```
python load_rte_nonuni,py
```
hsp_1d_two_materials
```
python load rte_sigma.py
```
hsp_1d_ani_eps_1_g5, hsp_1d_ani_eps_zpzz1_g5_bl and hsp_1d_ani_eps_1_g5_bl
```
python load_rte_aniso.py
```
hsp_1d_ani_eps_zpzz1_g5_bl
```
python load_rte_aniso_nonuni.py
```
hsp_1d_nonl__eps_1_bl and hsp_1d_nonl_eps_1
```
python load_rte_nonl.py
```
hsp_1d_nonl_eps_zpzz1 and hsp_1d_nonl_eps_zpzz1_bl
```
python load_rte_nonl_epsi.py
```
hsp_2d_bl_eps1, hsp_2d_analytic
```
python load_rte_2D.py
```
hsp_2d_bl
```
python rg_2D_hsp_load.py
```
