# amsSim

Adaptive Multilevel Splitting (AMS) simulation tools for rare-event option pricing and path generation in continuous-time models.  
Core routines are implemented in **C++ (Rcpp/RcppArmadillo)** for speed.

---

> **Status**: under active development; preparing for CRAN submission.

---

## Features

- Fast Monte-Carlo path simulation:
  - Black–Scholes (exact)
  - Heston variants (Euler / Milstein / QE Andersen 2008)
- Adaptive Multilevel Splitting estimator for rare events
- Clean R interface with vectorised inputs
- Minimal dependencies (Rcpp, RcppArmadillo)

---

## Installation

Install the development version from GitHub:

# Option A: via remotes
install.packages("remotes")
remotes::install_github("riccardogozzo/amsSim")

# Option B: via devtools
# install.packages("devtools")
devtools::install_github("riccardogozzo/amsSim")

# Option C: via pak (fast)
# install.packages("pak", repos = "https://r-lib.github.io/p/pak/dev/")
pak::pak("riccardogozzo/amsSim")

---

## Quick start

library(amsSim)

# Black–Scholes toy run (fast)
set.seed(1)
res <- simulate_AMS(1, n = 500, t = 1, p = 252, r = 0.03, sigma = 0.2, S0 = 1, rho = NULL)
str(res)

# AMS example (small, <5s)
set.seed(1)
out <- AMS(model = 2, type = 3, funz = 1, n = 500, t = 1, p = 252, r = 0.03,
             sigma = 0.2, rho = -0.5, S0 = 1, rim = 0, Lmax = 0.5, strike = 1.3, K = 200)
str(out)

---

## Main functions

# simulate_AMS(model, n, t, p, r, sigma, S0, rho = NULL, rim = 0L, v0 = 0.04)
  Returns simulated paths:
  model 1 (Black–Scholes): matrix/list with S of size n x (p - rim + 1)
  Heston models (2–4): list with S, V

# function_AMS_Cpp(S_paths, option, funz, strike, r, sigma, time)
  Builds the AMS score matrix from given paths.

# AMS(model, type, funz, n, t, p, r, sigma, S0, rho = NULL, rim = 0L, v0 = 0.04, Lmax = 0, strike = 1, K = 1L)
  Runs the adaptive splitting loop and returns list(price, std).

---

## System Requirements

- **R** version >= 4.0  
- **Rcpp** (>= 1.0.0), **RcppArmadillo** (>= 0.11)  
- Compilers:  
  - macOS: Xcode command line tools (clang++)  
  - Linux: g++ (>= 9), make, and BLAS/LAPACK libraries  
  - Windows: Rtools (>= 4.0)  

---

## Contributing

Contributions are welcome!  
If you wish to contribute, please:  

1. Fork the repository on GitHub.  
2. Create a new branch for your feature/bugfix.  
3. Add clear documentation and, if possible, unit tests.  
4. Submit a pull request.  

## License

This package is released under the **MIT License**.  
See the file [LICENSE](LICENSE) for details.  

---

## Citation

If you use **amsSim** in your research, please cite it as follows:

Gozzo, R. (2025). amsSim: Monte Carlo and Adaptive Multilevel Splitting Methods in R.
R package version 0.1.0.
Available at: https://github.com/riccardogozzo/amsSim




