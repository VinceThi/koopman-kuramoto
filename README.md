# koopman-kuramoto

Code and symbolic derivations associated with the article:

> **"Kuramoto meets Koopman: Constants of motion, symmetries, and network motifs"**  
> Vincent Thibeault, Benjamin Claveau, Antoine Allard, and Patrick Desrosiers

arXiv: https://arxiv.org/abs/2504.06248
Zenodo DOI: 10.5281/zenodo.20599776

## Overview

This repository provides the computational and symbolic framework supporting the results presented in the associated article. The work develops a Koopman operator formulation of the Kuramoto model, identifying classes of constants of motion, associated symmetries, and network motifs that constrain the dynamics.


## Repository Structure

The project is organized to facilitate the exploration of both analytical derivations and numerical experiments:

- **`symbolic/`**: Mathematica and Matlab scripts for the analytical derivation of symmetries and the proof of theorem 1
- **`dynamics/`**: Implementations of the Kuramoto dynamics, the constants of motion and the partial integration
- **`graphs/`**: Utilities for generating and analyzing network topologies and motifs  
- **`simulations/`**: Scripts for numerical verification of constants of motion and symmetries  
- **`plots/`**: Code used to generate figures in the manuscript  
- **`tests/`**: Unit tests ensuring consistency of the implementations  

## Citation

If this repository contributes to your research, please cite:

```bibtex
@article{thibeault2025kuramoto,
  title={Kuramoto meets Koopman: Constants of motion, symmetries, and network motifs},
  author={Thibeault, Vincent and Claveau, Benjamin and Allard, Antoine and Desrosiers, Patrick},
  journal={arXiv preprint arXiv:2504.06248},
  year={2025}
}
```

## Affiliation

This work was conducted within the [Dynamica Research Lab](https://dynamicalab.github.io/) at Université Laval.

## License

This project is released under the MIT License. See the LICENSE file for details.
