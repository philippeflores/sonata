# SONATA: damped ellipSe decOmpositioN for bivAriaTe signAls

A python implementation of the SONATA (damped ellipSe decOmpositioN for bivAriaTe signAls) algorithm

The code comes in addition of the paper submitted to SSP 2025:
> Flores, P., Flamant, J., Amblard, P.-O., Le Bihan, N., 2025. Damped ellipse decomposition for bivariate signals.

## Requirements

Code has been tested for Python 3.9.16. Several packages are required:

- [NumPy](http://www.numpy.org)
- [SciPy](https://www.scipy.org)
- [Matplotlib](http://matplotlib.org)
- [Bispy](https://bispy.readthedocs.io)

## Overview

This repository contains several Python notebooks to generate damped ellipse mixture, and reconstruct them using the SONATA procedure presented in the paper.

- **main_example** contains a routine example showing the whole pipeline of SONATA.
- **main_ringdown** contains an application of SONATA to ringdown gravitational wave data analysis.
