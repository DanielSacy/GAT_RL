# GAT_RL

GAT_RL implements a Graph Attention Network (GAT)–based reinforcement‑learning solver for the Capacitated Vehicle Routing Problem (CVRP). This code originates from a dissertation project and had an article presented at ICORES 2025 and unifies recent advances in learned routing heuristics into a single end‑to‑end framework. At its core, GAT_RL combines the attention‑based encoder–decoder architecture introduced by Kool et al. in “Attention, Learn to Solve Routing Problems!” with the residual edge‑graph attention enhancements proposed by Lei et al. (2021) [ARXIV](https://arxiv.org/abs/2105.02730). Training relies on the DiCE estimator (Differentiable Monte Carlo Estimator) to eliminate dual‑actor overhead while preserving solution quality.

Features
Attention‑based GAT encoder with residual edge features
DiCE estimator for low‑variance, single‑actor policy gradients
Flexible decoding strategies (greedy, sampling, beam search)
Synthetic and CVRPLIB benchmark support
Near‑optimal solutions with inference in seconds
Installation
bash
Copy
git clone https://github.com/DanielSacy/GAT_RL.git
cd GAT_RL
pip install -r requirements.txt
Quickstart
Generate synthetic CVRP data
bash
Copy
python generate_data.py --problem CVRP --num_nodes 50 --seed 42 --output data/cvrp50.pkl
Train
bash
Copy
python train.py \
  --dataset data/cvrp50.pkl \
  --model gat \
  --estimator dice \
  --epochs 100 \
  --batch_size 128
Evaluate
bash
Copy
python evaluate.py \
  --checkpoint checkpoints/cvrp50_dice.pt \
  --dataset data/cvrp50.pkl \
  --decode greedy
Citation
If you use GAT_RL in your research, please cite:

bibtex
Copy
@inproceedings{turn8search0,
  title={Integrating Machine Learning and Optimisation to Solve the Capacitated Vehicle Routing Problem},
  author={Pedrozo, Daniel Antunes and Gupta, Prateek and Meira, Jorge Augusto and Silva, Fabiano},
  booktitle={ICORES},
  year={2025}
}

@inproceedings{turn9search0,
  title={Attention, Learn to Solve Routing Problems!},
  author={Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle={ICLR},
  year={2019}
}

@article{turn11search1,
  title={Solve routing problems with a residual edge-graph attention neural network},
  author={Lei, Kun and Guo, Peng and Wang, Yi and Wu, Xiao and Zhao, Wenchao},
  journal={arXiv preprint arXiv:2105.02730},
  year={2021}
}

@article{turn10search4,
  title={DiCE: The Infinitely Differentiable Monte Carlo Estimator},
  author={Foerster, Jakob and Farquhar, Gregory and Al-Shedivat, Maruan and Rockt{\"a}schel, Tim and Xing, Eric P and Whiteson, Shimon},
  journal={arXiv preprint arXiv:1802.05098},
  year={2018}
}
License
This project is released under the MIT License
