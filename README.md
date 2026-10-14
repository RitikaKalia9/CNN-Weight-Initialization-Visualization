<div align="center">

# 🧠 CNN Initialization & Visualization Lab

### Understand how CNNs learn — from weight initialization to visual explanations

**PyTorch · CIFAR-10 · Captum · Computer Vision**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-2E8B57)
![Focus](https://img.shields.io/badge/Focus-CNN%20Interpretability-7B61FF)

*A practical lab on keeping signals stable during training and investigating what a trained CNN responds to.*

</div>

---

## 📌 About the Project

A Convolutional Neural Network (CNN) can be affected by **how its weights are initialized** and can be difficult to interpret after training. This lab investigates both questions using a small six-convolution-layer CNN and the **CIFAR-10** image dataset.

The work is divided into two experiments:

| Experiment | Goal |
|---|---|
| **01 — Weight Initialization** | Implement LeCun, Xavier, and He initialization from scratch, compare them with PyTorch, and measure activation scales and short-run training behaviour. |
| **02 — CNN Visualization** | Visualize learned filters and feature maps, then use attribution methods to investigate which image pixels or regions support a class score. |

> **Core idea:** Good initialization helps maintain useful signal scales through a deep network. Visualization methods provide evidence about model behaviour, but no single heatmap proves exactly how a model makes a decision.

## 🧪 Experiment 01 — Weight Initialization

### 1. Fan-in and fan-out

For a convolutional layer with kernel height \(K_H\), kernel width \(K_W\), \(C_{\text{in}}\) input channels, and \(C_{\text{out}}\) output channels:

\[
\boxed{\text{fan-in}=K_HK_WC_{\text{in}}}
\]

\[
\boxed{\text{fan-out}=K_HK_WC_{\text{out}}}
\]

For a linear layer, fan-in and fan-out correspond to its input and output feature counts.

**Example:** A \(5\times5\) convolution with 16 input channels and 32 output channels has:

\[
\text{fan-in}=5\times5\times16=400
\]

\[
\text{fan-out}=5\times5\times32=800
\]

With a bias for each output channel, the total parameter count is:

\[
25(16)(32)+32=\boxed{12{,}832}
\]

Fan-in is **not** the total parameter count; it counts the weights contributing to one output value.

### 2. The three initialization rules

Let \(n_{\text{in}}\) denote fan-in and \(n_{\text{out}}\) denote fan-out.

| Initializer | Target weight variance | Normal distribution standard deviation | Uniform distribution bound |
|---|---|---|---|
| **LeCun** | \(\displaystyle \frac{1}{n_{\text{in}}}\) | \(\displaystyle \sqrt{\frac{1}{n_{\text{in}}}}\) | \(\displaystyle a=\sqrt{\frac{3}{n_{\text{in}}}}\) |
| **Xavier / Glorot** | \(\displaystyle \frac{2}{n_{\text{in}}+n_{\text{out}}}\) | \(\displaystyle \sqrt{\frac{2}{n_{\text{in}}+n_{\text{out}}}}\) | \(\displaystyle a=\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\) |
| **He / Kaiming** | \(\displaystyle \frac{2}{n_{\text{in}}}\) | \(\displaystyle \sqrt{\frac{2}{n_{\text{in}}}}\) | \(\displaystyle a=\sqrt{\frac{6}{n_{\text{in}}}}\) |

The normal forms use zero-mean weights with the listed standard deviation. The uniform forms sample from \(U(-a,a)\).

#### Why are the formulas different?

- **LeCun:** uses variance \(1/n_{\text{in}}\) to maintain the scale of a linear signal under the stated assumptions.
- **Xavier:** balances forward- and backward-pass scale using both fan-in and fan-out.
- **He:** uses variance \(2/n_{\text{in}}\) to compensate for the reduction in mean-square signal caused by ReLU activations.

For a uniform random variable:

\[
\operatorname{Var}(U(-a,a))=\frac{a^2}{3}
\]

Therefore, to obtain a target variance \(v\):

\[
\boxed{a=\sqrt{3v}}
\]

### 3. Implementations in PyTorch

The manual implements the formulas from scratch and compares them with `torch.nn.init`.

```python
import math
import torch
import torch.nn as nn

def fans(w):
    fan_in = w[0].numel()
    fan_out = w.shape[0] * w[0, 0].numel()
    return fan_in, fan_out

def fill(w, var, uniform=False):
    if uniform:
        a = math.sqrt(3 * var)
        w.uniform_(-a, a)
    else:
        w.normal_(0, math.sqrt(var))

def lecun_init(w, uniform=False):
    fan_in, _ = fans(w)
    fill(w, 1 / fan_in, uniform)

def xavier_init(w, uniform=False):
    fan_in, fan_out = fans(w)
    fill(w, 2 / (fan_in + fan_out), uniform)

def he_init(w, uniform=False):
    fan_in, _ = fans(w)
    fill(w, 2 / fan_in, uniform)
```

Equivalent PyTorch initializers used in the lab:

```python
# LeCun
nn.init.kaiming_normal_(weight, nonlinearity="linear")
nn.init.kaiming_uniform_(weight, nonlinearity="linear")

# Xavier / Glorot
nn.init.xavier_normal_(weight)
nn.init.xavier_uniform_(weight)

# He / Kaiming for ReLU
nn.init.kaiming_normal_(weight, nonlinearity="relu")
nn.init.kaiming_uniform_(weight, nonlinearity="relu")
```

**Note:** PyTorch calls He initialization *Kaiming initialization*. In this lab, LeCun initialization is obtained through the Kaiming initializer with `nonlinearity="linear"`. A newly created `nn.Conv2d` layer should not be assumed to use the lab's explicit He initialization.

### 4. What the lab measures

The CNN contains six \(3\times3\) convolutional layers with ReLU activations and no BatchNorm in the initial comparison. The lab records the activation root-mean-square (RMS):

\[
\boxed{\operatorname{RMS}(h)=\sqrt{\operatorname{mean}(h^2)}}
\]

It compares the PyTorch default, LeCun, Xavier, He, and a fixed standard deviation of \(0.01\).

The manual's expected pattern is that **He keeps the activation scale near 1** across the tested layers, while the fixed \(0.01\) initialization causes the activations to shrink substantially. A short CIFAR-10 training comparison then demonstrates how initialization can affect early learning.

## 🔍 Experiment 02 — Visualizing a Trained CNN

The CNN is trained with He initialization. The lab then examines its filters, activations, and class attributions using `torchvision` and Captum.

| Technique | What it investigates | Limitation to remember |
|---|---|---|
| **First-layer filters** | Patterns represented by the first convolutional kernels | Deeper kernels operate on feature channels, not directly on RGB pixels |
| **Feature maps** | Which channels activate for a particular image | Results depend on the selected image and layer |
| **Saliency** | Sensitivity of the class score to input pixels | Can be noisy; sensitivity is not the same as contribution |
| **Integrated Gradients** | Pixel attribution relative to a baseline image | Depends on the baseline and uses repeated gradient evaluations |
| **Guided Backpropagation** | Fine-grained positive-gradient visualizations | Visually sharp maps are not necessarily faithful |
| **Grad-CAM** | Coarse image regions supporting a class | Resolution depends on the selected convolutional layer |
| **Guided Grad-CAM** | Fine detail within a class-specific Grad-CAM region | Inherits Guided Backpropagation's caveat |
| **Occlusion** | How a class score changes when an image patch is hidden | More computationally expensive; depends on patch size and fill value |

### Grad-CAM: key equations

Let \(S^c\) be the score for class \(c\), and let \(A^k\) be feature map \(k\) with \(Z\) spatial positions.

The importance weight for feature map \(k\) is:

\[
\boxed{\alpha_k^c=\frac{1}{Z}\sum_{i,j}
\frac{\partial S^c}{\partial A_{ij}^k}}
\]

The Grad-CAM map is:

\[
\boxed{L_{\mathrm{Grad-CAM}}^c=
\operatorname{ReLU}\left(\sum_k\alpha_k^cA^k\right)}
\]

The ReLU retains positive evidence for the target class. In this lab, the last convolutional block produces a coarse \(8\times8\) map, which is interpolated to the input's \(32\times32\) resolution for display.

### Occlusion: the basic idea

Occlusion hides a patch of the image and measures the change in the target-class score. If hiding a region lowers the score, that region may support the prediction.

The manual's \(32\times32\) example uses a \(5\times5\) patch moved in steps of two pixels, requiring **196 forward passes**. Smaller patches give finer spatial detail; larger patches test broader regions.

## 🛠️ Tech Stack

- **Python** — implementation
- **PyTorch** — CNN, training, gradients, and initialization
- **torchvision** — CIFAR-10 dataset and filter grids
- **Captum** — attribution and interpretability methods
- **Matplotlib** — plots and visualizations
- **CIFAR-10** — 10-class image dataset

## ⚙️ Setup and Usage

### Requirements

- Python 3.9+
- Jupyter Notebook or Google Colab
- CPU or optional CUDA-compatible GPU

### Install dependencies

```bash
pip install torch torchvision matplotlib captum
```

### Run the lab

1. Open a notebook and copy the code cells from the accompanying lab manual in order.
2. Run the setup and initialization sections first.
3. Let `torchvision` download CIFAR-10 when requested.
4. Complete Experiment 1 before Experiment 2, because the second experiment reuses functions and variables from the first.
5. Inspect the printed statistics, accuracy, filter grids, feature maps, attribution maps, and runtime comparisons.

> **Reproducibility:** The manual fixes the PyTorch random seed to `0`. Exact timings and accuracy may still vary with hardware, software versions, and runtime conditions. Focus on the trends rather than expecting every number to match exactly.

## 🧩 Further Experiments

- Calculate fan-in, fan-out, and parameter count for \(3\times3\) and \(1\times1\) convolutions.
- Compare LeCun and Xavier on `nn.Linear(100, 100)`.
- Generate Grad-CAM and Guided Backpropagation for the second-most-likely class.
- Add BatchNorm after convolution layers and compare activation scales.
- Compare \(3\times3\), \(5\times5\), and \(10\times10\) occlusion windows.

## 🎯 Learning Outcomes

By completing this lab, you will learn to:

- Derive and calculate fan-in and fan-out.
- Implement LeCun, Xavier, and He initialization in normal and uniform forms.
- Compare custom initializers with PyTorch's built-in methods.
- Observe how initialization affects activation scale and early training.
- Visualize convolution filters and intermediate feature maps.
- Generate and interpret common CNN attribution maps.
- Discuss the strengths and limitations of gradient-based and perturbation-based explanations.

## 📚 References

- [PyTorch — Weight initialization](https://pytorch.org/docs/stable/nn.init.html)
- [Torchvision documentation](https://pytorch.org/vision/stable/index.html)
- [Captum documentation](https://captum.ai/docs/)

---

<div align="center">

**Made for learning, experimentation, and understanding CNN behaviour.**

</div>
