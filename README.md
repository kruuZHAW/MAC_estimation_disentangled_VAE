<div align="center">   
  
# Advanced Collision Risk Estimation in Terminal Manoeuvring Areas using a Disentangled Variational Autoencoder for Uncertainty Quantification 

## Description
This package provides a disentangled VAE architectrue to assess the risk of collision between aircraft procedures using uncertainty quantification. This project relies on [traffic](https://traffic-viz.github.io/), [Pytorch-Lightning](https://www.pytorchlightning.ai/) and [OpenTurns](http://openturns.github.io/openturns/latest/contents.html) libraries. This repository reproduces the works in the paper [Advanced Collision Risk Estimation in Terminal Manoeuvring Areas using a Disentangled Variational Autoencoder for Uncertainty Quantification 
](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4581022)

## Install package
 ```bash
conda create -n mac_estimation -c conda-forge python=3.10 traffic
conda activate mac_estimation
pip install .
```

## Training of the disentangled VAE
Navigate to `deep_traffic_generation` folder and put both of the training datasets in `data/training_datasets` (on for each air traffic procedure) using the traffic format from the [traffic](https://traffic-viz.github.io/) library. Make sure that every aircraft trajectories have the same number of points.    
 
 ```bash
cd deep_traffic_generation
python tcvae_pairs_disent.py --data_path [traffic 1] [traffic 2] --n_samples 20000 --prior factorized_vampprior --n_components 2 --encoding_dim 10 --tc_coef 20 --kld_coef 20 --h_dims 64 64 64 64 64 --lr 0.001 --lrstep 200 --lrgamma 0.5 --gradient_clip_val 0.5 --batch_size 1000 --features track groundspeed altitude timedelta
```

## MAC estimation
Navigate to `prob_estimate` folder, and choose either `monte_carlo` or `subset_simulation`. Then run either `simu_runs_mc.py` or `simu_run_ss` by making sure that the `load_TCVAE()` method uses the adequate path paramters for the training datasets and the trained VAE. 

## Sensitivity analysis
Navigate to `prob_estimate/sensitivity_analysis` and run `sobol.py`. Make sure that the `load_TCVAE()` method uses the right path paramters for the training datasets and the trained VAE.
