<h1 align="center">D3L Data Discovery Framework</h1>
<p align="center">Similarity-based data discovery in data lakes</p>

<p align="center">
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This is the home of D3l data discovery framework: an approximate implementation of the [ICDE 2020 paper](https://arxiv.org/pdf/2011.10427.pdf) with the same name.

## Getting started

This is an approximate implementaion of the [D3L research paper](https://arxiv.org/pdf/2011.10427.pdf) published at ICDE 2020.
The implementation is approximate because not all notions proposed in the paper are transfered to code. The most notable differences are mentioned below:
* The indexing evidence for numerical data is different from the one presented in the paper. In this package, numerical columns are transformed to their density-based histograms and indexed under a random projection LSH index.
* The distance aggregation function (Equation 3 from the paper) is not yet implemented. In fact, the aggregation function is customizable. During testing, a simple average of distances has proven comparabel to the level reported in the paper.
* The package uses similarity scores (between 0 and 1) instead of distances, as described in the paper.

## Installation

You'll need Python 3.6.x to use this package.

```
pip install git+https://github.com/alex-bogatu/d3l
```

### Installing from a specific release

You may wish to install a specific release. To do this, you can run:

```
pip install git+https://github.com/alex-bogatu/d3l@{tag|branch}
```

Substitute a specific branch name or tag in place of `{tag|branch}`.

## Usage

See [here]() for an example notebook.

## Contributing

All contributions must conform to [PEP-8](https://www.python.org/dev/peps/pep-0008/) and code style [Black](https://github.com/psf/black).
This package adopts `numpy` style docstrings for in-code documentation. See the [numpy GitHub](https://github.com/numpy/numpy) repo for examples.