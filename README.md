# ConfoNess

ConfoNess, short for Conformational Non-Equilibrium Steady State, is a
Python tool for modeling chemical reactions involving conformational
changes in molecules.  It can be used to model general systems of
chemical reactions, but it has been written primarily to model
molecular conformational dynamics driven by consumption of energy
molecules such as ATP.  It provides the following functions:

* Construct kinetic equations from a set of chemical reactions 
* Solve the time evolution of the kinetic equations, given initial conditions
* Find the steady state concentrations 
* Parameterize kinetic rate constants by fitting to the steady state experimental data
* Plot the reaction network

## Prerequisites

ConfoNess depends on [networkx](https://networkx.github.io/) and [sympy](https://www.sympy.org/en/index.html).  You can install these using anaconda:

```
conda install -c anaconda networkx
conda update sympy
```

## Civil matters

ConfoNess is free open source software.  NO WARRANTY, Use AS IS.

Copyright (C) 2018-2020 Huafeng Xu

If you use ConfoNess in a published work, please cite

* Huafeng Xu, ATP-driven non-equilibrium activation of kinase clients by the molecular chaperone Hsp90, 2020, [doi: 10.1101/2020.05.10.087577](https://doi.org/10.1101/2020.05.10.087577)
* Huafeng Xu, Cochaperones enable Hsp70 to use ATP energy to stabilize native proteins out of the folding equilibrium, 2018, [Sci. Rep. 8, 13213](https://doi.org/10.1038/s41598-018-31641-w).

ConfoNess was developed as the computational tool to enable these studies. 

## How to use

You can find usage examples in [examples.py](https://github.com/forcefield/ConfoNess/blob/master/examples.py), [hsp90.py](https://github.com/forcefield/ConfoNess/blob/master/hsp90.py), and the Jupyter notebook [hsp90.ipynb](https://github.com/forcefield/ConfoNess/blob/master/hsp90.ipynb).
