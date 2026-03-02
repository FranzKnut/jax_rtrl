# ODE Cell — RTRL Formulas

## State update (Euler)

The hidden state is evolved with a fixed-step Euler integrator:

$$h_{t+1} = h_t + f(h_t, x_t;\, \theta) \cdot \Delta t$$

---

## Jacobian traces

The carry stores two Jacobian traces:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $J^\theta_t$ | $H \times \ldots_\theta$ | $\partial h_t / \partial \theta$ |
| $J^x_t$ | $H \times I$ | $\partial h_t / \partial x_t$ |

where $H$ = `num_units`, $I$ = input dim, $\ldots_\theta$ = parameter shape.

### Immediate Jacobians (current step)

$$\frac{\partial f}{\partial \theta},\quad \frac{\partial f}{\partial h},\quad \frac{\partial f}{\partial x}
\quad \leftarrow \quad \texttt{jax.jacrev(ode, argnums=[0,1,2])}(\theta, h_t, x_t)$$

### State-to-state Jacobian

$$\frac{\partial h_{t+1}}{\partial h_t} = I + \frac{\partial f}{\partial h} \cdot \Delta t$$

### RTRL trace update

$$J^\theta_{t+1} = \frac{\partial h_{t+1}}{\partial h_t} \cdot J^\theta_t + \frac{\partial f}{\partial \theta} \cdot \Delta t$$

$$J^x_{t+1} = \frac{\partial h_{t+1}}{\partial h_t} \cdot J^x_t + \frac{\partial f}{\partial x} \cdot \Delta t$$

The matrix-trace product uses an einsum to contract over the **first axis** of $J^\theta_t$:

$$\left(\frac{\partial h_{t+1}}{\partial h_t} \cdot J^\theta_t\right)_{i,\ldots} = \sum_j \left(\frac{\partial h_{t+1}}{\partial h_t}\right)_{ij} \cdot J^\theta_{t,\,j,\ldots}$$

i.e. `einsum("ij,j...->i...", dh_dh, jp)`.

### SNAP-0 (zeroth-order approximation)

Drops the recurrent term entirely, keeping only the immediate Jacobian:

$$J^\theta_{t+1} \approx \frac{\partial f}{\partial \theta} \cdot \Delta t, \qquad J^x_{t+1} \approx \frac{\partial f}{\partial x} \cdot \Delta t$$

---

## Gradient computation

Given a loss gradient $\delta_t = \partial \mathcal{L} / \partial h_t$:

### RTRL

$$\frac{\partial \mathcal{L}}{\partial \theta} = \delta_t \cdot J^\theta_t \qquad \text{(row vector × matrix)}$$

### RFLO (first-order approximation)

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \delta_t^{\top} \odot J^\theta_t$$

### Input gradient

$$\frac{\partial \mathcal{L}}{\partial x_t} = \sum_h \delta_{t,h} \cdot J^x_{t,\,h,\,\cdot} = \texttt{einsum}(\text{"...h,...hi->...i"},\, \delta_t,\, J^x_t)$$
