# Introduction
This project apply PINN to steady state of radiative transfer equation(RTE). The main difficulty comes from
the significantly varying magnitude of the Knusen number. 


# Usage of code
In following example, user needs to input some parameters. The epsi is the Knudsen number, bc_loss_weight is the weight of boundary loss
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
hsp_2d_bl_eps1, hsp_2d_analytic
```
python load_rte_2D.py
```
hsp_2d_bl
```
python rg_2D_hsp_load.py
```
