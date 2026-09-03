# The Ion Neural Network (INN): The First Model Class of Neurotransmitter Ion Receptor Glial Endocannabinoid Network Models

## Discrete-Ion Neural Computation

**nirgen** is a Python/PyTorch implementation of the Neurotransmitter Ion Receptor Glial Endocannabinoid Network Models an **Ionic Neural Network (INN)**: a neural computation framework in which computation is represented through the discrete movement, accumulation, partitioning, and release of ionic resources within a bounded cellular system.

Rather than representing a neuron primarily as a sequence of matrix multiplications and nonlinear activation functions, NIRGEN represents a computational unit as a constrained physical system.

The fundamental computational resource is an **ion**.

An INN contains an extracellular compartment (`ecto`), an intracellular compartment (`endo`), input and output patches, receptor inventories, vesicle inventories, and a set of fixed stoichiometric and geometric constraints. Computation occurs through a deterministic sequence of discrete state transitions.

The repository currently implements the core **single-cell, six-operation INN forward-pass architecture** described in the accompanying NIRGEN work.

---

## Conceptual Overview

A conventional neural network generally represents a computation as something resembling

$$
y = f(Wx+b).
$$

NIRGEN instead represents computation through changes in a bounded state:

$$
S^{(t)}
\rightarrow
S^{(t+1)}
$$

where the state consists of discrete ionic populations distributed across compartments and patches.

The model is therefore closer to a constrained dynamical system than to a conventional feed-forward neural network.

At a high level:

```text
                    AMBIENT ION POOL
                           │
                           ▼
                 ┌───────────────────┐
                 │   ECTO COMPARTMENT │
                 │                   │
 Input ─────────►│  Input Patches    │
                 │       │           │
                 └───────┼───────────┘
                         │
                  receptor activation
                         │
                         ▼
                 ┌───────────────────┐
                 │  ENDO COMPARTMENT │
                 │                   │
                 │ intracellular     │
                 │ ion population     │
                 └───────┬───────────┘
                         │
                       diffusion
                         │
                         ▼
                 ┌───────────────────┐
                 │   OUTPUT PATCHES  │
                 │                   │
                 │ vesicle fusion    │
                 └───────┬───────────┘
                         │
                         ▼
                    ECTO RELEASE
                         │
                         ▼
                       OUTPUT
```

The important distinction is that the network does not treat the intermediate values simply as abstract activations. They are populations of a conserved resource subject to capacity, availability, and stoichiometric constraints.

---

# The INN Computational Model

The current implementation uses a single-cell architecture with:

* `n` input patches
* `m` output patches
* extracellular (`ecto`) ionic populations
* intracellular (`endo`) ionic populations
* receptor inventories on input patches
* vesicle inventories on output patches
* fixed receptor thresholds
* fixed translocation quanta
* fixed vesicle thresholds
* fixed extrusion quanta
* fixed diffusion weights
* α and β allocation coefficients
* a membrane-side weighting parameter `sigma`

The repository currently provides multiple ion/species configurations through the internal species registry.

A model is constructed with:

```python
from nirgen.inn.model import INN
```

and instantiated as:

```python
model = INN(
    species="Na",
    n=4,
    m=2
)
```

The exact available species and device presets are defined by the package registry.

---

# Six-Operation Forward Pass

The current implementation follows a six-operation schedule.

## 1. Input perturbation

The input modifies the extracellular ionic state of the input patches.

The implementation converts the supplied input into a perturbation of the extracellular state using a Nernst-based transformation.

Conceptually:

$$
x
\rightarrow
S_{\mathrm{ecto}}^{(1)}.
$$

The input therefore represents a perturbation to an ionic environment rather than simply becoming a vector of neuron activations.

---

## 2. Stoichiometric splitting

The current ionic populations are divided into flux and activation sub-pools.

For the input side:

$$
F_{\mathrm{ecto}}
=
\left\lfloor
\alpha S_{\mathrm{ecto}}
\right\rfloor
$$

and

$$
A_{\mathrm{ecto}}
=
S_{\mathrm{ecto}}-F_{\mathrm{ecto}}.
$$

The same principle is applied to the intracellular population using `beta`.

The two sub-pools have different computational roles:

* **activation ions** provide the material required to trigger receptors;
* **flux ions** provide material available for translocation.

---

## 3. Receptor activation and translocation

Each input patch has a trainable receptor inventory.

The number of receptors that can actually open is constrained simultaneously by:

1. the number of receptors available;
2. the activation ions available;
3. the flux ions available;
4. the available receiving capacity.

The effective receptor count is therefore determined by a minimum over these constraints.

Conceptually:

$$
r_e =
\min
\left\{
r,\,
\left\lfloor\frac{A}{\{r\}}\right\rfloor,\,
\left\lfloor
\frac{\sigma F_{\mathrm{ecto}}
+(1-\sigma)F_{\mathrm{endo}}}
{|[r]|}
\right\rfloor,\,
\left\lfloor
\frac{\text{available capacity}}
{|[r]|}
\right\rfloor
\right\}.
$$

When receptors open, ions are transferred across the membrane according to the receptor's translocation quantum.

The implementation also tracks the activation material associated with receptor operation.

---

## 4. Inward diffusion

After receptor activation, the resulting intracellular and extracellular populations are distributed across the output patches.

The current implementation uses fixed area/fraction-based diffusion weights.

For example, the intracellular population entering an output patch is proportional to its output-patch weight:

$$
S_{l_o,\mathrm{endo}}
=
w_{l_o}
\sum_{l_i}
S_{l_i,\mathrm{endo}}.
$$

The diffusion weights are not learned by Adam.

They are part of the fixed configuration of the INN.

---

## 5. Output stoichiometric splitting

The intracellular population at each output patch is again divided into:

* a flux sub-pool;
* an activation sub-pool.

This provides the resources required for vesicle activation and subsequent extrusion.

---

## 6. Vesicle fusion and exocytosis

Each output patch has a trainable vesicle inventory.

The number of vesicles that can fuse is constrained by:

1. the available vesicles;
2. the activation population required to trigger fusion;
3. the available flux material and extracellular receiving capacity.

Conceptually:

$$
V_e =
\min
\left\{
V,\,
\left\lfloor
\frac{A_{\mathrm{endo}}}{\{V\}}
\right\rfloor,\,
\left\lfloor
\frac{
\min(F_{\mathrm{endo}},
C_{\mathrm{ecto}}-S_{\mathrm{ecto}})
}
{[V]}
\right\rfloor
\right\}.
$$

A successful fusion event transfers the extrusion quantum back into the extracellular compartment.

The final output is then read from the resulting extracellular state.

---

# What Is Learned?

The current implementation deliberately has a very small trainable parameter space.

The trainable quantities are the **receptor and vesicle inventories**:

$$
r_{l_i}
$$

for input patches, and

$$
V_{l_o}
$$

for output patches.

The underlying implementation stores continuous relaxation variables:

```python
model.r_tilde
model.v_tilde
```

while the actual forward computation uses discrete inventories derived from those values.

This permits gradient-based optimization despite the discrete nature of the physical model.

The forward computation therefore operates approximately as:

```text
continuous parameters
        │
        ▼
 straight-through floor
        │
        ▼
 discrete receptor/vesicle inventories
        │
        ▼
 ionic state transitions
        │
        ▼
      output
        │
        ▼
       loss
        │
        ▼
      Adam
        │
        └──────────► updated continuous parameters
```

---

# Basic Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/drmylesgarveylabs/nirgen.git
```

The repository is currently a lightweight PyTorch package.

For development:

```bash
git clone https://github.com/drmylesgarveylabs/nirgen.git
cd nirgen

pip install -e .
```

---

# Basic Model Construction

A minimal INN can be created as follows:

```python
import torch
from nirgen.inn.model import INN

model = INN(
    species="Na",
    n=4,
    m=2
)

print(model)
```

Here:

* `species` selects the ionic/device configuration;
* `n=4` creates four input patches;
* `m=2` creates two output patches.

The model is a standard PyTorch `nn.Module`, so it can participate in normal PyTorch optimization workflows.

---

# Running Inference

Inputs are supplied as a PyTorch tensor.

For four input patches and a batch of eight observations:

```python
import torch
from nirgen.inn.model import INN

model = INN(
    species="Na",
    n=4,
    m=2
)

x = torch.randn(8, 4)

with torch.no_grad():
    y = model(x)

print(y.shape)
print(y)
```

The expected output has one value for each output patch:

```text
(batch_size, number_of_output_patches)
```

For this example:

```text
(8, 2)
```

The model performs the complete six-operation ionic computation internally.

---

# Training with Adam

Because `INN` is implemented as a PyTorch module, the trainable receptor and vesicle inventories can be optimized using standard PyTorch optimizers.

The following example uses **Adam**.

```python
import torch
from torch import nn

from nirgen.inn.model import INN

# Construct the ionic neural network
model = INN(
    species="Na",
    n=4,
    m=2
)

# Adam optimizes the continuous relaxation parameters
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()

# Example training data
x = torch.randn(256, 4)
target = torch.randn(256, 2)

for epoch in range(1000):

    optimizer.zero_grad()

    # Ionic forward pass
    prediction = model(x)

    # Compare ionic output with target
    loss = criterion(prediction, target)

    # Backpropagation through the straight-through
    # discrete-state machinery
    loss.backward()

    # Update receptor/vesicle relaxation parameters
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"epoch={epoch:04d} "
            f"loss={loss.item():.6f}"
        )
```

The important point is that Adam does **not** directly turn the receptor inventories into arbitrary floating-point neural-network weights.

The implementation maintains continuous relaxation variables and converts them to discrete inventories during the forward computation.

Thus:

```python
model.r_tilde
model.v_tilde
```

are the quantities Adam updates, while the ionic computation uses their discrete counterparts.

---

# Inspecting the Learned Inventories

After training, the learned receptor and vesicle inventories can be inspected directly.

```python
with torch.no_grad():
    receptors = model.r_int()
    vesicles = model.v_int()

print("Receptor inventories:")
print(receptors)

print("Vesicle inventories:")
print(vesicles)
```

These values represent the discrete number of computational devices available at each patch.

The distinction is important:

```text
r_tilde / v_tilde
    continuous optimization variables

        ↓

r_int / v_int
    discrete physical inventories

        ↓

ionic state transitions
```

---

# Training with a Dataset

The same approach can be used with a normal PyTorch `DataLoader`.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nirgen.inn.model import INN

x = torch.randn(5000, 4)
y = torch.randn(5000, 2)

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

model = INN(
    species="Na",
    n=4,
    m=2
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()

for epoch in range(100):

    running_loss = 0.0

    for x_batch, y_batch in loader:

        optimizer.zero_grad()

        prediction = model(x_batch)

        loss = criterion(
            prediction,
            y_batch
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"epoch={epoch:03d} "
        f"loss={running_loss / len(loader):.6f}"
    )
```

Nothing in the training loop requires a conventional linear layer, convolution, attention mechanism, or activation function.

The model itself defines the forward computation.

---

# Regularization

The model also exposes a regularization function for receptor and vesicle inventories:

```python
regularization = model.reg(
    lam_r=1e-4,
    lam_v=1e-4
)
```

This can be incorporated into the training objective:

```python
prediction = model(x)

data_loss = criterion(prediction, target)

reg_loss = model.reg(
    lam_r=1e-4,
    lam_v=1e-4
)

loss = data_loss + reg_loss

loss.backward()
optimizer.step()
```

This provides a mechanism for penalizing larger receptor and vesicle inventories.

---

# Custom Diffusion Geometry

The default model uses equal patch fractions.

The model can also be constructed using explicit diffusion fractions:

```python
model = INN(
    species="Na",
    n=4,
    m=2,
    w_in_fracs=[0.25, 0.25],
    w_out_fracs=[0.25, 0.25, 0.25, 0.25]
)
```

The input and output fraction vectors must each sum to one.

For example:

```python
sum([0.25, 0.25]) == 0.5
```

would **not** be valid for an output vector with two patches.

Instead:

```python
w_in_fracs=[0.5, 0.5]
```

is valid.

The fractions determine how the global ionic populations are apportioned across patches and how inward diffusion distributes ionic populations.

The diffusion weights are fixed configuration parameters rather than learned weights.

---

# Readout Modes

The model supports two readout modes.

The default is:

```python
readout=1
```

which uses the Nernst-based extracellular readout.

A second mode is available:

```python
readout=2
```

which uses the signed extracellular/intracellular difference.

For example:

```python
model = INN(
    species="Na",
    n=4,
    m=2,
    readout=2
)
```

This produces a two-dimensional output corresponding to the two output patches.

---

# Calibration Modes

The implementation currently supports fixed and pool-based calibration modes.

The default is:

```python
cal_mode="fixed"
```

For example:

```python
model = INN(
    species="Na",
    n=4,
    m=2,
    cal_mode="fixed"
)
```

The alternative pool-based mode is:

```python
model = INN(
    species="Na",
    n=4,
    m=2,
    cal_mode="pool"
)
```

In fixed mode, receptor and vesicle type parameters are obtained from the configured device type.

In pool mode, these quantities are derived from fractions of the available ionic pools.

These modes are part of the current implementation and are intended to provide alternative ways of initializing/calibrating the ionic machinery.

---

# Inspecting Conservation

The package provides a conservation audit:

```python
x = torch.randn(4, 4)

audit = model.conservation_check(x)

for operation, total in audit.items():
    print(operation, total)
```

The audit reports the total tracked ionic population at each stage of the six-operation computation.

This is useful when experimenting with:

* new species;
* new device configurations;
* new patch geometries;
* new stoichiometric parameters;
* alternative calibration settings.

Conservation is a central design principle of the ionic computation.

---

# A Complete Minimal Example

The following script constructs an INN, trains it with Adam, and evaluates it.

```python
import torch
from torch import nn

from nirgen.inn.model import INN


# --------------------------------------------------
# 1. Data
# --------------------------------------------------

torch.manual_seed(42)

x = torch.randn(1024, 4)
target = torch.randn(1024, 2)


# --------------------------------------------------
# 2. Ionic Neural Network
# --------------------------------------------------

model = INN(
    species="Na",
    n=4,
    m=2
)


# --------------------------------------------------
# 3. Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()


# --------------------------------------------------
# 4. Training
# --------------------------------------------------

for epoch in range(500):

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(
        output,
        target
    )

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:

        print(
            f"epoch={epoch:04d} "
            f"loss={loss.item():.6f}"
        )


# --------------------------------------------------
# 5. Inspect learned physical inventories
# --------------------------------------------------

print("\nReceptor inventories:")
print(model.r_int())

print("\nVesicle inventories:")
print(model.v_int())


# --------------------------------------------------
# 6. Inference
# --------------------------------------------------

test_x = torch.randn(10, 4)

with torch.no_grad():
    prediction = model(test_x)

print("\nPrediction:")
print(prediction)
```

---

# Why This Is Different From a Conventional Neural Network

An INN should not be interpreted simply as a conventional neural network with biological terminology substituted for standard layers.

The computational primitive is different.

A conventional network might contain:

```text
input
  ↓
matrix multiplication
  ↓
activation function
  ↓
matrix multiplication
  ↓
output
```

An INN instead performs:

```text
input perturbation
  ↓
ionic partition
  ↓
receptor activation
  ↓
ion translocation
  ↓
diffusion
  ↓
ionic partition
  ↓
vesicle activation
  ↓
ion extrusion
  ↓
readout
```

The constraints are therefore part of the computation.

For example, increasing the receptor inventory does not necessarily increase the output. If another constraint is binding, additional receptors have no effect.

Likewise, increasing the vesicle inventory does not necessarily increase output if the available activation ions, flux ions, or extracellular capacity are limiting.

This produces a computational system in which **resource availability is itself part of the state of the machine**.

---

# Current Scope

The current repository implements the core single-cell INN computation.

In particular, it currently provides:

* discrete ionic state variables;
* bounded extracellular and intracellular compartments;
* input perturbation;
* stoichiometric splitting;
* receptor activation;
* receptor translocation;
* inward diffusion;
* output stoichiometric splitting;
* vesicle activation;
* vesicle exocytosis;
* Nernst-based readout;
* discrete receptor and vesicle inventories;
* straight-through optimization of discrete inventories;
* PyTorch/Adam-compatible training;
* inventory regularization;
* conservation auditing;
* configurable patch fractions;
* configurable calibration modes.

The implementation should be regarded as a research implementation and experimental computational framework rather than as a validated biological simulation.

The goal is to investigate whether a computational architecture grounded in discrete resource dynamics, bounded state spaces, stoichiometric constraints, and event-driven state transitions can serve as a useful alternative neural-computation paradigm.

---

# Citation

If you use NIRGEN in academic work, please cite the associated paper/preprint:

> Garvey, M. D. *NIRGEN: Discrete Transmembrane Particle Dynamics and Bounded State-Space Computation.*

See the accompanying manuscript for the formal mathematical definition of the model, state space, transition operators, diffusion functions, activation constraints, training formulation, and conservation framework.

---

# License

See the repository license for the terms governing use and redistribution.

---

# Repository

Source code:

https://github.com/drmylesgarveylabs/nirgen
