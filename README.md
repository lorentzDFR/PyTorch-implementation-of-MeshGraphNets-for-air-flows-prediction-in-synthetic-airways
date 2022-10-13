# PyTorch-implementation-of-MeshGraphNets-for-air-flows-prediction-in-synthetic-airways


The code in this repository is my PyTorch version of "**MeshGraphNets**", mainly written using *PyTorch Geometric* library. The model architecture was presented in 
"*Learning to Simulate Complex Physics with Graph Networks*" (2020) (https://arxiv.org/abs/2002.09405) and "*Learning Mesh-Based Simulation with Graph Networks*" (ICML 2021) (https://arxiv.org/abs/2010.03409) by Google DeepMind.

<p align="center">
<img src="https://user-images.githubusercontent.com/82932496/195567601-247b775f-671d-433a-b4e0-3beb876aa1ce.png" width="550" >

The final goal of my project was the development of a simulator based on graph networks, capable of describing air flows inside synthetic airways. The project followed several steps. At first we have domain production
<p align="center">

<img src="https://user-images.githubusercontent.com/82932496/195568561-e3b991c6-9c3e-4552-8d58-14aa560c91f6.png" width="180" >

and air flows simulations with numerical methods
<p align="center">
<img src="https://user-images.githubusercontent.com/82932496/195573608-ea7c9a81-bbc9-4f05-a40f-02d01e6917e6.png" width="550" >


After that i extracted point clouds from the domains and created a radius-based connectivity
<p align="center">
<img src="https://user-images.githubusercontent.com/82932496/195570622-4dbec4bb-7f26-477e-9469-dad2c7754a37.png" width="480" >
<img src="https://user-images.githubusercontent.com/82932496/195570950-3a2d228e-b748-4291-a66b-39d66f5caee6.png" width="305" >



Every simulation has its own slightly different geometry and a fixed point cloud associated, with extracted node pressure and velocity components across time

<p align="center">
<img src="https://user-images.githubusercontent.com/82932496/195571579-52a5882a-a7de-4da8-b2d5-781fa9b2819d.png" width="650" >


The dataset is composed of 1000 simulations of 500 time steps each. I used a 700/100/200 train/validation/test split. The way the model is trained is very similar to the procedures followed in the above mentioned papers. As i write, the model is tested only on one-step predictions and is not used in an autoregressive fashion.
An illustration of the one step prediction on a test simulation is here illustrated:

<p align="center">
<img src="https://user-images.githubusercontent.com/82932496/195576824-3bf856c2-eceb-4cfd-97d3-151e2b5a538b.gif" width="700" >
