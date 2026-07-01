# ☕ Fish in the Percolator

*A simple Fisher forecasting demonstration using coffee cooling and DerivKit.*

## Motivation

Many scientific problems involve estimating model parameters from noisy data.

Suppose we repeatedly measure the temperature of a cup of coffee as it cools. 
From these measurements we would like to answer questions such as

- How hot was the coffee when it was poured?
- How quickly does it cool?
- How accurately can we determine these quantities?
- Which parameters are strongly correlated?
- How would our uncertainties change if we collected more measurements?

These are exactly the kinds of questions addressed by **Fisher forecasting**.

This repository provides a small, self-contained demonstration of Fisher forecasting 
using **Newton's law of cooling**, with forecasts computed using **DerivKit**.

Although the example is intentionally simple, the same workflow scales to much
larger scientific models with dozens of parameters.

---

# The model

We describe the temperature of the coffee using Newton's law of cooling

$$T(t)=T_{\rm room}+\left(T_0-T_{\rm room}\right)e^{-t/\tau}$$

where

- $T_0$ is the initial coffee temperature,
- $T_{\rm room}$ is the ambient room temperature,
- $\tau$ is the cooling time,
- $t$ is the elapsed time.

The temperature approaches the room temperature exponentially.

In the simplest example we estimate only two parameters

$$\theta=(T_0,\tau)$$

while keeping the room temperature fixed.

Later we introduce additional nuisance parameters to illustrate 
larger Fisher forecasts and triangle plots.

---

# Synthetic observations

Instead of using real data, we generate synthetic measurements.

For a chosen fiducial parameter vector

$$\theta_0$$

we evaluate the cooling curve

$$T(t;\theta_0)$$

and add Gaussian measurement noise

$$d_i=T(t_i;\theta_0)+\mathcal{N}(0,\sigma_T)$$

This produces a mock experiment that we can use to study parameter constraints.

---

# Fisher forecasting

Suppose our data vector is

$$\mathbf{d}(\theta)$$

The Fisher matrix is

$$F_{ij}=\frac{\partial\mathbf{d}}{\partial\theta_i}^{\rm T}C^{-1}\frac{\partial\mathbf{d}}{\partial\theta_j}$$

where

- $C$ is the data covariance matrix,
- $\partial\mathbf d/\partial\theta_i$ is the derivative of the model with respect to parameter $i$.

The Fisher matrix approximates the local curvature of the likelihood around the fiducial model.

The corresponding parameter covariance is simply

$$\Sigma=F^{-1}$$

From this covariance we obtain

- parameter uncertainties,
- parameter correlations,
- confidence ellipses,
- triangle plots.

---

# Why derivatives?

Computing the Fisher matrix requires derivatives of the model with respect to every parameter.

For this small toy model the derivatives could be derived analytically.

However, realistic scientific models often involve

- numerical simulations,
- differential equation solvers,
- Monte Carlo methods,
- external software,
- hundreds of parameters.

In these situations computing derivatives accurately and efficiently becomes one of the primary challenges.

This is exactly the problem solved by **DerivKit**.

---

# Why coffee?

The coffee example is intentionally familiar.

Everyone understands that

- coffee starts hot,
- coffee cools,
- measurements contain noise.

This allows us to focus entirely on

- parameter estimation,
- uncertainty quantification,
- Fisher forecasting,

without introducing domain-specific knowledge.

Exactly the same workflow applies to

- cosmology,
- astronomy,
- climate science,
- biology,
- engineering,
- economics,

or any scientific model that predicts observables from parameters.

---

# Repository structure

```
src/
    model.py
        Coffee cooling models.

    data.py
        Synthetic data generation.

    fisher.py
        DerivKit Fisher utilities.

    plots.py
        Plotting helpers.

scripts/
    make_coffee_data.py
        Generates noisy observations.

    plot_two_param_forecast.py
        Produces a two-parameter Fisher forecast.

    plot_triangle_forecast.py
        Produces a larger Fisher triangle plot.
```

---

# Running the demo

Install

```bash
pip install -e .
```

Generate synthetic observations

```bash
percolator-data
```

Create the two-parameter Fisher forecast

```bash
percolator-two-param
```

Create the larger Fisher triangle plot

```bash
percolator-triangle
```

---

# Interpreting the results

## Cooling curve

The first figure shows

- noisy temperature measurements,
- the underlying cooling model.

This represents the experiment.

---

## Two-parameter forecast

The second figure shows confidence ellipses for

- initial temperature,
- cooling time.

The ellipse illustrates

- parameter uncertainties,
- parameter correlations.

A narrow ellipse corresponds to well-constrained parameters.

A tilted ellipse indicates that increasing one parameter can be partially compensated by changing another.

---

## Triangle plot

The final figure extends the model to additional parameters.

Each diagonal panel shows the marginalized distribution of one parameter.

Each off-diagonal panel shows the joint constraints for a pair of parameters.

Circular contours indicate nearly independent parameters.

Tilted ellipses indicate correlated parameters.

Long thin ellipses indicate strong parameter degeneracies.

---

# Beyond this toy model

This repository is intentionally small.

The purpose is not to build the most realistic model of coffee cooling.

Instead, it illustrates the complete Fisher forecasting workflow

```
Model
        ↓
Synthetic observations
        ↓
Model derivatives
        ↓
Fisher matrix
        ↓
Covariance matrix
        ↓
Confidence contours
        ↓
Scientific interpretation
```

The exact same workflow is used in many areas of modern computational science, 
where DerivKit automates the derivative calculations that make Fisher forecasting possible.