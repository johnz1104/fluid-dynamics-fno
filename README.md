# Fluid Dynamics Neural Operator

This repository implements Fourier Neural Operator (FNO) surrogates for terminal-state
prediction in stochastic fluid PDEs. It includes pseudo-spectral solvers and training
pipelines for the 1D Burgers and 2D Navier–Stokes equations.

## Stochastic PDE models

### 1D stochastic Burgers equation

```text
du/dt + u du/dx = nu d²u/dx² + sigma dW/dt
```

This is a one-dimensional model of nonlinear advection, diffusion, and random
forcing. The pseudo-spectral solver uses an integrating factor for diffusion and
Heun's method for the nonlinear term.

### 2D stochastic Navier–Stokes equation

```text
d(omega)/dt + J(psi, omega) = nu Laplacian(omega) + curl(f)
```

The two-dimensional solver uses vorticity and streamfunction, so incompressibility
is built into the formulation. It uses Crank–Nicolson for diffusion and
Adams–Bashforth for advection.

For both problems, random forcing is smooth in space and generated in Fourier space.
The FNO approximates the terminal-state map
$(u_0,\bar{\xi})\mapsto u(\cdot,T)$ from the initial field, time-averaged forcing,
and spatial coordinates. Its layers combine truncated Fourier multipliers for
non-local coupling with pointwise residual transforms for local channel mixing.

The time average is an important limitation: two different forcing histories can
have the same average but produce different final states. The model therefore cannot
recover every effect of the full forcing history.

## Results

### Burgers benchmark

The recorded Burgers experiment used 1,000 generated samples and trained for 50
epochs on a CPU.

| Metric | Error |
|---|---:|
| Mean relative L2 | 7.30% |
| Median relative L2 | 7.03% |
| 95th-percentile relative L2 | 10.55% |
| Ensemble mean field | 6.39% |
| Ensemble variance field | 1.82% |

![Burgers prediction examples](val_burgers_realizations.png)

![Burgers ensemble statistics](val_burgers_ensemble_stats.png)

## Project layout

```text
fno.py          1D and 2D Fourier neural operators
solvers.py      Stochastic PDE solvers and data generation
train.py        Datasets, losses, training, and evaluation
main.py         Command-line entry point
validation.py   Validation metrics and plots
visualization.py
```

## Run the stochastic models

Generate data:

```bash
python main.py --pde burgers --mode generate --num_samples 1000
python main.py --pde ns --mode generate --num_samples 500
```

Train a model:

```bash
python main.py --pde burgers --mode train --epochs 100 --batch_size 32
python main.py --pde ns --mode train --epochs 200 --batch_size 16
```

Evaluate a trained model:

```bash
python validation.py \
  --pde burgers \
  --model_path checkpoints/best_model.pt \
  --data_path burgers_test.npy

python validation.py \
  --pde ns \
  --model_path checkpoints/ns_final.pt \
  --data_path ns_test.npy
```

## Documentation

- [`IMPLEMENTATION.md`](IMPLEMENTATION.md): original stochastic-PDE implementation
  guide.
