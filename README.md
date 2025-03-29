# jax-rtrl

Fast implementation of Real-Time Recurrent Learning (RTRL) [1] and Random Feedback Local Online (RFLO) Learning [2] for Continuous Time Recurrent Neural Networks [3] and Linear Recurrent Units [4] in flax.

Work in progress!

> If you found this repository useful in your research, please consider citing the following paper:

```bibtex
@article{lemmel2024,
  title = {Real-Time Recurrent Reinforcement Learning},
  author = {Lemmel, Julian and Grosu, Radu},
  year = {2024},
  month = mar,
  url = {http://arxiv.org/abs/2311.04830},
  doi = {10.48550/arXiv.2311.04830},
  urldate = {2024-04-03},
}
```


[1] R. J. Williams and D. Zipser, “A Learning Algorithm for Continually Running Fully Recurrent Neural Networks,” Neural Computation, vol. 1, no. 2, pp. 270–280, Jun. 1989, doi: 10.1162/neco.1989.1.2.270.

[2] J. M. Murray, “Local online learning in recurrent networks with random feedback,” eLife, vol. 8, p. e43299, May 2019, doi: 10.7554/eLife.43299.

[3] K. Funahashi and Y. Nakamura, “Approximation of dynamical systems by continuous time recurrent neural networks,” Neural Networks, vol. 6, no. 6, pp. 801–806, Jan. 1993, doi: 10.1016/S0893-6080(05)80125-X.

[4] N. Zucchet, R. Meier, S. Schug, A. Mujika, and J. Sacramento, “Online learning of long-range dependencies,” in Thirty-seventh Conference on Neural Information Processing Systems, Nov. 2023.
