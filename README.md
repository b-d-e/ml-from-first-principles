# Machine Learning from First Principles

[![Actions Status](https://github.com/b-d-e/ml-from-first-principles/workflows/CI/badge.svg)](https://github.com/b-d-e/ml-from-first-principles/actions) [![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)](https://github.com/b-d-e/ml-from-first-principles)

<!-- [![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link] -->


Implementing Machine Learning algorithms from first principles, without relying on Python libraries.

![Alt](misc/banner.jpeg "A robot learning Linear Algebra, generated by DALLE.")

## 📋 To Do

This will incrementally build up functionality for ML algorithms - initially this will be naively and slow. Over time, I hope to iteratively improve, benchmarking against numpy, scikit-learn, etc.

### Linear Algebra Fundamentals:
On branch [issues/1/linear-algebra](https://github.com/b-d-e/ml-from-first-principles/tree/issues/1/linear-algebra)

Implement:

- [x] Vector Operations
    - [x] Addition & Subtraction
    - [x] Scalar Multiplication
    - [x] Vector Multiplication (Dot Product)
    - [x] Unit Tests
- [x] Generalise to Matrices
    - [x] Addition & Subtraction
    - [x] Scalar Multiplication
    - [x] Matrix Multiplication (Dot Product)
    - [x] Unit Tests
- [x] Matrix Transposition
    - [x] Unit Tests

> _mlfp.linear_algebra is currently *incredibly* slow compared to numpy. no optimisation attempts made yet:_
```
Matrix creation timing (numpy): 0.000746000005165115
Matrix creation timing (mlfp): 2.462499833200127e-05
Matrix addition timing (numpy): 0.0003061659954255447
Matrix addition timing (mlfp): 0.05240800000319723
Matrix multiplication timing (numpy): 0.016511209003510885
Matrix multiplication timing (mlfp): 11.73682175000431
```

## 🛣️ Roadmap

As a start, I'll look to create a simple dense MNIST classifier. Roughly, these are the steps I will look to sequentially implement:
1. simple linear algebra / core numpy equiv
2. basic densely connected network, represented with weights and biases
3. Implement common activation functions
4. forward pass - for single layer, multi-layer, whole network
5. loss function (start with cross entropy)
6. batched losses
7. backpropogation - chain rule to backprop errors, calculating new w&b
8. optimisation algo for gradient descent
9. data loader
10. training loop, combining above
11. evaluation

## 🔧 Installation

<!-- ```bash
python -m pip install .
``` -->

From source:
```bash
git clone https://github.com/b-d-e/ml-from-first-principles
cd ml-from-first-principles
python -m pip install .
```

## Usage
```python
import mlfp
```

<!-- ## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute. -->

## License

Distributed under the terms of the [GNU General Public License](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/b-d-e/ml-from-first-principles/workflows/CI/badge.svg
[actions-link]:             https://github.com/b-d-e/ml-from-first-principles/actions
<!-- [pypi-link]:                https://pypi.org/project/Machine Learning from First Principles/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/Machine Learning from First Principles
[pypi-version]:             https://img.shields.io/pypi/v/Machine Learning from First Principles -->
<!-- prettier-ignore-end -->
