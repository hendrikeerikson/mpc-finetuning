# Experiments in private neural network fine-tuning using multi-party computation.

* `approximations.nb` contains Wolfram Mathematica code for deriving polynomial approximations of elementary functions, used in floating-point protocols.
* `secrec/` contains SecreC code for recreating benchmarks of different fine-tuning scenarios.
* `finetune.ipynb` is a Jupyter Notebook which implements the last-layer fine-tuning scenario in plaintext using PyTorch.
* `python/` contains code for generating secret shares of pre-trained model
weights and executing the SecreC code with Sharemind MPC. A license and
installation of Sharemind MPC is required.

