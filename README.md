# The Ion Neural Network (INN): The First Model Class of Neurotransmitter Ion Receptor Glial Endocannabinoid Network Models

## Discrete-Ion Neural Computation

**nirgen** is a Python/PyTorch implementation of the Neurotransmitter Ion Receptor Glial Endocannabinoid Network Models an **Ionic Neural Network (INN)**: a neural computation framework in which computation is represented through the discrete movement, accumulation, partitioning, and release of ionic resources within a bounded cellular system.

The fundamental computational unit is an **ion**.

Rather than representing computation exclusively through unconstrained floating-point transformations such as

```text
y = f(Wx + b)
```

an INN represents a computational state as populations of ions distributed across an extracellular environment, intracellular environment, membrane/input patches, and output patches.

Computation occurs through a sequence of constrained state transitions involving:

```text
Input perturbation
        ↓
Stoichiometric partition
        ↓
Receptor activation
        ↓
Ion translocation
        ↓
Intracellular diffusion
        ↓
Output partition
        ↓
Vesicle activation
        ↓
Exocytosis
        ↓
Output readout
```

The current repository implements the core single-cell INN architecture using PyTorch.

---

# Model Architecture

The computational system can be visualized as:

```text
                         AMBIENT ION POOL
                                │
                                │
                                ▼
              ┌─────────────────────────────────┐
              │        EXTRACELLULAR            │
              │             ECTO                │
              │                                 │
 Input ──────►│   Input Patch 1                 │
              │        │                        │
              │        │ receptor               │
              │        ▼                        │
              │   Input Patch 2                 │
              │        │                        │
              │        ▼                        │
              └────────┼────────────────────────┘
                       │
                 ion translocation
                       │
                       ▼
              ┌─────────────────────────────────┐
              │       INTRACELLULAR             │
              │            ENDO                 │
              │                                 │
              │      ionic population           │
              │           │                     │
              │           │ diffusion           │
              │           ▼                     │
              │     Output Patch 1              │
              │     Output Patch 2              │
              │           │                     │
              └───────────┼─────────────────────┘
                          │
                    vesicle fusion
                          │
                          ▼
              ┌─────────────────────────────────┐
              │        EXTRACELLULAR            │
              │          RELEASE                │
              └──────────────┬──────────────────┘
                             │
                             ▼
                           OUTPUT
```

For a publication-quality illustration, see `docs/inn_architecture.svg`.

The SVG can be embedded directly into this README:

```html
<p align="center">
  <img
    src="docs/inn_architecture.svg"
    alt="NIRGEN Ionic Neural Network architecture"
    width="900"
  >
</p>
```

GitHub will render the SVG directly from the repository.

---

# Core State

The INN maintains ionic populations across bounded compartments.

The principal state variables are:

* extracellular ionic population (`ecto`);
* intracellular ionic population (`endo`);
* input-patch populations;
* output-patch populations;
* boundary/engaged ionic material associated with receptor operation.

The model is designed around discrete state transitions and constrained resource availability.

The computational resource is not an abstract activation value. It is an ionic population subject to:

* availability;
* compartment capacity;
* receptor stoichiometry;
* vesicle stoichiometry;
* translocation constraints;
* diffusion;
* extracellular receiving capacity.

---

# The Six-Operation Forward Pass

The current implementation executes the INN computation as a fixed six-operation schedule.

## 1. Input Perturbation

The external input perturbs the extracellular ionic population of the input patches.

The input is therefore interpreted as an environmental perturbation of the ionic system.

```text
input
  ↓
extracellular ionic state
```

---

## 2. Stoichiometric Split — Input

Each input patch divides its extracellular population into two sub-pools.

The flux pool is:

```text
F_ecto = floor(alpha × S_ecto)
```

and the activation pool is:

```text
A_ecto = S_ecto - F_ecto
```

The activation pool supplies ions required to trigger receptor opening.

The flux pool supplies ions available for translocation.

---

## 3. Receptor Activation and Translocation

Each input patch contains a receptor inventory.

The actual number of receptors that open during a forward pass is constrained by multiple simultaneous requirements:

```text
available receptors
        +
activation-ion availability
        +
flux availability
        +
receiving capacity
```

The effective receptor count is therefore the minimum of these constraints.

The receptor type specifies:

```text
rth = activation threshold
rq  = translocation quantum
```

where:

* `rth` is the number of ions required to activate one receptor;
* `rq` is the number of ions translocated per receptor opening.

The receptor inventory itself is trainable.

---

## 4. Inward Diffusion

Following receptor activity, ionic populations are distributed from the input patches to the output patches.

The current implementation uses fixed diffusion fractions.

These fractions determine how ionic material is distributed across the output geometry.

Diffusion weights are **not learned parameters**.

They are part of the fixed configuration of the computational system.

---

## 5. Stoichiometric Split — Output

At the output patches, the intracellular population is again divided into activation and flux sub-pools.

The output-side split uses:

```text
F_endo = floor(beta × S_endo)
```

and

```text
A_endo = S_endo - F_endo
```

The activation pool is used to trigger vesicle fusion.

The flux pool provides material available for extrusion.

---

## 6. Vesicle Fusion and Exocytosis

Each output patch contains a vesicle inventory.

The actual number of vesicles that fuse is constrained by:

```text
available vesicles
        +
activation-ion availability
        +
available extrusion material
        +
extracellular receiving capacity
```

The vesicle type specifies:

```text
vth = fusion threshold
vq  = extrusion quantum
```

where:

* `vth` is the number of ions required to activate one vesicle;
* `vq` is the number of ions released per fusion event.

The resulting extracellular state is then used to produce the model output.

---

# Biophysical Registry

The repository contains a registry of species-specific and device-specific parameters.

The registry is defined in:

```text
nirgen/inn/registry.py
```

The current implementation defines the following global constants:

```python
RHO = 2.30
D_POOL_NM = 100_000.0

ION1 = math.floor(
    RHO * math.pi * D_POOL_NM**2 / 4
)

PHI_T = 25.7
Z_ION = 1

NERNST_EPS_IONS = 1e-6
NERNST_EXP_CLIP = 20.0

INVENTORY_JITTER = 2.0
INVENTORY_JITTER_REL = 0.05

RTH_MIN, RQ_MIN, VTH_MIN, VQ_MIN = 5, 2, 5, 2
```

These values provide the base physical/numerical configuration used by the model.

---

# Device Types

The current registry defines four biological/device presets:

| Device type | Receptor threshold | Receptor quantum | Vesicle threshold | Vesicle quantum |
| ----------- | -----------------: | ---------------: | ----------------: | --------------: |
| Human       |                  8 |                4 |                 4 |               4 |
| Mouse       |                  8 |                4 |                 4 |               4 |
| Drosophila  |                  4 |                4 |                 2 |               4 |
| C. elegans  |                  4 |                2 |                 2 |               2 |

These values are represented in the registry as:

```python
DEVICE_TYPES = {
    "Human": {
        "rth": 8,
        "rq": 4,
        "vth": 4,
        "vq": 4
    },

    "Mouse": {
        "rth": 8,
        "rq": 4,
        "vth": 4,
        "vq": 4
    },

    "Drosophila": {
        "rth": 4,
        "rq": 4,
        "vth": 2,
        "vq": 4
    },

    "C. elegans": {
        "rth": 4,
        "rq": 2,
        "vth": 2,
        "vq": 2
    },
}
```

The device-type parameters describe the fixed behavior of receptor and vesicle devices.

They are distinct from the **inventories** of those devices.

---

# Species Configurations

The current implementation provides four species configurations:

```python
SPECIES = {
    "Human": ...,
    "Mouse": ...,
    "Drosophila": ...,
    "C. elegans": ...,
}
```

Each species configuration specifies the geometry, ionic density, compartment capacities, resting populations, diffusion parameters, allocation coefficients, and calibration fractions used by the model.

For example, the Human configuration contains:

```python
"Human": dict(
    d_endo=20_000.0,
    d_ecto=20_080.0,

    rho=RHO,
    f=0.05,

    C_endo=723_000_000,
    C_ecto=5_800_000,

    S_Bendo_0=36_000_000,
    S_Becto_0=290_000,

    D=36_290_000,

    delta=2e-3,
    gamma=8e-3,

    alpha=0.5,
    beta=0.5,
    sigma=1.0,

    f_rth=0.10,
    f_rq=0.05,
    f_vth=0.10,
    f_vq=0.05,

    two_cmp=True
)
```

The corresponding configurations for Mouse, Drosophila, and C. elegans are defined in the same registry.

---

# Calibration Modes

Two calibration modes are currently available:

```python
CAL_MODES = (
    "pool",
    "fixed"
)
```

## Fixed

```python
cal_mode="fixed"
```

uses the registered device-type parameters.

## Pool

```python
cal_mode="pool"
```

derives device parameters from fractions of the available ionic pools.

This allows the model to be calibrated relative to the scale of the ionic state rather than exclusively through the fixed biological/device presets.

---

# Type Presets

The implementation also provides simplified type presets:

```python
TYPE_PRESETS = {
    "paper": None,

    "low": {
        "rth": 2,
        "rq": 2,
        "vth": 2,
        "vq": 2
    },

    "mid": {
        "rth": 4,
        "rq": 4,
        "vth": 4,
        "vq": 4
    },

    "high": {
        "rth": 8,
        "rq": 8,
        "vth": 8,
        "vq": 8
    },
}
```

These presets are useful for experimentation with different stoichiometric scales.

---

# Creating an INN

The model is a standard PyTorch module.

```python
import torch

from nirgen.inn.model import INN

model = INN(
    species="Human",
    n=4,
    m=2
)
```

The model can then be used like any other PyTorch module:

```python
x = torch.randn(32, 4)

y = model(x)

print(y.shape)
```

For four input patches and two output patches, the resulting output is:

```text
torch.Size([32, 2])
```

---

# Training with Adam

The trainable quantities in the current implementation are the continuous relaxation variables associated with receptor and vesicle inventories.

This permits the discrete ionic computation to be optimized using gradient-based methods.

The simplest training configuration uses Adam:

```python
import torch
from torch import nn

from nirgen.inn.model import INN


model = INN(
    species="Human",
    n=4,
    m=2
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()
```

A complete training loop is:

```python
x = torch.randn(1024, 4)
target = torch.randn(1024, 2)

for epoch in range(500):

    optimizer.zero_grad()

    prediction = model(x)

    loss = criterion(
        prediction,
        target
    )

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"epoch={epoch:04d} "
            f"loss={loss.item():.6f}"
        )
```

Adam updates the continuous trainable parameters.

The forward pass subsequently maps those continuous quantities into the discrete receptor and vesicle inventories used by the ionic computation.

---

# Inspecting Learned Inventories

The learned continuous parameters can be inspected directly:

```python
print(model.r_tilde)
print(model.v_tilde)
```

The corresponding discrete inventories can be obtained through the model's integer inventory functions:

```python
with torch.no_grad():

    receptor_inventory = model.r_int()
    vesicle_inventory = model.v_int()

print(receptor_inventory)
print(vesicle_inventory)
```

This distinction is fundamental to the current implementation:

```text
continuous relaxation
        │
        ▼
straight-through discrete conversion
        │
        ▼
integer receptor / vesicle inventory
        │
        ▼
ionic state transition
```

---

# Regularization

The model provides an inventory regularization term.

```python
reg_loss = model.reg(
    lam_r=1e-4,
    lam_v=1e-4
)
```

It can be incorporated into the training objective:

```python
prediction = model(x)

data_loss = criterion(
    prediction,
    target
)

reg_loss = model.reg(
    lam_r=1e-4,
    lam_v=1e-4
)

loss = data_loss + reg_loss

loss.backward()
optimizer.step()
```

This allows the optimization objective to penalize larger receptor and vesicle inventories.

---

# Inference

Once training is complete:

```python
model.eval()

test_x = torch.randn(10, 4)

with torch.no_grad():

    prediction = model(test_x)

print(prediction)
```

The model performs the complete ionic forward pass internally.

No separate neural-network layers need to be constructed by the user.

---

# Conservation Audit

The implementation includes a conservation audit:

```python
audit = model.conservation_check(x)

for operation, total in audit.items():

    print(
        operation,
        total
    )
```

This is intended to make it possible to inspect the ionic population as it progresses through the computational schedule.

Conservation is a central design principle of NIRGEN.

When modifying the model, registry, stoichiometric parameters, or diffusion mechanism, the conservation audit can be used to identify unintended changes in the total tracked ionic resource.

---

# Readout

The model supports two output readout modes.

The Nernst-based readout is the default:

```python
model = INN(
    species="Human",
    n=4,
    m=2,
    readout=1
)
```

An alternative signed compartment-difference readout is:

```python
model = INN(
    species="Human",
    n=4,
    m=2,
    readout=2
)
```

The first mode uses the ionic concentration relationship underlying the Nernst equation.

The second mode returns the signed difference between the extracellular and intracellular populations.

---

# Custom Patch Geometry

The diffusion geometry can be configured using patch fractions.

For example:

```python
model = INN(
    species="Human",
    n=4,
    m=2,

    w_in_fracs=[
        0.50,
        0.50
    ],

    w_out_fracs=[
        0.25,
        0.25,
        0.25,
        0.25
    ]
)
```

The fractions determine how ionic populations are distributed across patches.

They are fixed configuration parameters rather than learned neural-network weights.

---

# Complete Example

```python
import torch
from torch import nn

from nirgen.inn.model import INN


# -----------------------------------------------
# Data
# -----------------------------------------------

torch.manual_seed(42)

x = torch.randn(1024, 4)
target = torch.randn(1024, 2)


# -----------------------------------------------
# Ionic Neural Network
# -----------------------------------------------

model = INN(
    species="Human",
    n=4,
    m=2
)


# -----------------------------------------------
# Adam optimizer
# -----------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()


# -----------------------------------------------
# Training
# -----------------------------------------------

for epoch in range(500):

    optimizer.zero_grad()

    output = model(x)

    data_loss = criterion(
        output,
        target
    )

    regularization = model.reg(
        lam_r=1e-4,
        lam_v=1e-4
    )

    loss = (
        data_loss
        + regularization
    )

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:

        print(
            f"epoch={epoch:04d} "
            f"loss={loss.item():.6f}"
        )


# -----------------------------------------------
# Learned inventories
# -----------------------------------------------

with torch.no_grad():

    receptors = model.r_int()
    vesicles = model.v_int()

print("\nReceptor inventories:")
print(receptors)

print("\nVesicle inventories:")
print(vesicles)


# -----------------------------------------------
# Inference
# -----------------------------------------------

model.eval()

test_x = torch.randn(10, 4)

with torch.no_grad():

    prediction = model(test_x)

print("\nPrediction:")
print(prediction)
```

---

# Design Philosophy

NIRGEN is intended to explore a different computational abstraction for neural systems.

The central question is not simply:

> How can a function approximate a target?

Instead, the architecture asks:

> Can computation be represented as the evolution of a bounded physical state subject to resource, capacity, stoichiometric, and conservation constraints?

The current INN implementation therefore makes several quantities that are normally hidden inside neural-network mathematics explicit:

```text
RESOURCE
    ions

STATE
    ionic populations

BOUNDARIES
    ecto / endo compartments

DEVICES
    receptors / vesicles

STOICHIOMETRY
    thresholds / quanta

CAPACITY
    finite compartment sizes

DYNAMICS
    discrete state transitions

TRANSPORT
    diffusion / translocation

LEARNING
    optimization of device inventories
```

The resulting system is not intended to be interpreted simply as a conventional neural network with biological terminology.

It is an experimental computational architecture based on discrete ionic state dynamics.

---

# Current Implementation Scope

This repository currently provides the core single-cell INN implementation, including:

* discrete ionic state variables;
* extracellular and intracellular compartments;
* bounded compartment capacities;
* input perturbation;
* α/β stoichiometric partitioning;
* receptor activation;
* receptor translocation;
* inward diffusion;
* output stoichiometric partitioning;
* vesicle activation;
* vesicle exocytosis;
* Nernst-based computation;
* discrete receptor inventories;
* discrete vesicle inventories;
* straight-through optimization;
* Adam-compatible training;
* inventory regularization;
* conservation auditing;
* species-specific configurations;
* biological/device presets;
* fixed and pool calibration modes;
* configurable patch fractions.

The repository should currently be considered a **research implementation / computational prototype**.

It is not a validated biophysical simulation of a biological neuron.

Its purpose is to provide an executable implementation of the NIRGEN computational abstraction and a foundation for experimentation with ionic neural computation.

---

# Repository Structure

A typical installation contains:

```text
nirgen/
│
├── nirgen/
│   └── inn/
│       ├── model.py
│       └── registry.py
│
├── docs/
│   └── inn_architecture.svg
│
├── tests/
│
├── pyproject.toml
└── README.md
```

The most important files for understanding the implementation are:

```text
nirgen/inn/model.py
```

The computational mechanics and PyTorch model.

```text
nirgen/inn/registry.py
```

The species, device, calibration, and stoichiometric registries.

```text
docs/inn_architecture.svg
```

The architectural illustration.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/drmylesgarveylabs/nirgen.git

cd nirgen
```

Install in editable mode:

```bash
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/drmylesgarveylabs/nirgen.git
```

---

# Status

NIRGEN is under active development.

The mathematical formulation and implementation are evolving together. The repository therefore intentionally exposes the underlying mechanisms rather than hiding them behind a high-level neural-network API.

Contributions, experiments, alternative species configurations, device configurations, diffusion mechanisms, and independent validation are welcome.

---

# Citation

If you use NIRGEN in academic work, please be so kind to cite the associated manuscript (which I will be publishing as a pre-print in a few days):

**Garvey, M. D. — Shadows of Consciousness: An Investigation into Ionic Neural Networks Using the Neurotransmitter Ion Receptor Glial Endocannabinoid Network (NIRGEN) Paradigm**

The manuscript contains the formal mathematical definition of the state space, transition operators, receptor and vesicle constraints, diffusion operators, readout mechanisms, and training formulation.

---

# License

See the repository license for terms governing use and redistribution.

---

# Repository

https://github.com/drmylesgarveylabs/nirgen
