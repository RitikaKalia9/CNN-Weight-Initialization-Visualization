CNN Weight Initialization and Visualization with PyTorch
A hands-on lab exploring how weight initialization affects CNN training and how to interpret a trained Convolutional Neural Network (CNN) using visualization and attribution techniques.
Overview
This lab uses a small CNN and the CIFAR-10 dataset to study two topics:
1. Weight initialization: Implement LeCun, Xavier (Glorot), and He (Kaiming) initialization from scratch, compare them with PyTorch's built-in initializers, and observe their effects on activation scales and short training runs.
2. CNN visualization: Train a CNN with He initialization and inspect its first-layer filters, feature maps, and class-related image regions using Captum attribution methods.
The examples are designed to run in a Jupyter Notebook or Google Colab on a CPU; a GPU is used automatically when available.
Features
Experiment 1 — Weight Initialization
- Calculate fan-in and fan-out for convolutional and linear layers.
- Implement LeCun, Xavier, and He initialization with normal and uniform distributions.
- Compare custom implementations with torch.nn.init.
- Track activation root-mean-square (RMS) values through a six-layer CNN.
- Compare short training runs using different initialization schemes.
Experiment 2 — Visualizing a Trained CNN
- Visualize first-layer convolution filters before and after training.
- Inspect early and late feature maps using forward hooks.
- Generate attribution maps with:
  - Saliency
  - Integrated Gradients
  - Guided Backpropagation
  - Grad-CAM
  - Guided Grad-CAM
  - Occlusion
- Measure the relative runtime of selected visualization methods.
Initialization Formulas
Let \(n_{in}\) be fan-in and \(n_{out}\) be fan-out.
Method	Weight variance
LeCun	\(1/n_{in}\)
Xavier / Glorot	\(2/(n_{in}+n_{out})\)
He / Kaiming	\(2/n_{in}\)


For a uniform distribution \(U(-a,a)\), use \(a=\sqrt{3\,\mathrm{Var}(w)}\) to obtain the target variance.
Key idea: Initialization affects the scale of signals and gradients as they pass through a network. He initialization is especially relevant to the lab's ReLU-based CNN because it compensates for the signal reduction associated with ReLU activations.
Requirements
- Python 3.9 or later
- Jupyter Notebook or Google Colab
- PyTorch
- torchvision
- matplotlib
- Captum
Installation
Install the required libraries:
pip install torch torchvision matplotlib captum
How to Run
1. Open the lab notebook in Jupyter or Google Colab.
2. Run the cells in order, starting with the setup and imports.
3. Allow the CIFAR-10 dataset to download when prompted by the code.
4. Complete Experiment 1 before starting Experiment 2, because Experiment 2 reuses functions, the CNN definition, and dataset variables created earlier.
5. Review the printed activation scales, test accuracy, filter grids, feature maps, attribution maps, and timing results.
The lab manual is a set of notebook-style code cells. If you are using the accompanying manual as a PDF/DOCX rather than a notebook, copy the code cells into a notebook in their original order.
Dataset
The experiments use CIFAR-10, a dataset of 32×32 colour images across 10 classes. The dataset is downloaded automatically by the provided torchvision code.
Methods at a Glance
Method	What it helps inspect	Main limitation noted in the lab
First-layer filters	Patterns learned by the first convolutional layer	Deeper filters act on feature channels, not directly on RGB pixels
Feature maps	A layer's activations for a particular input	Activations depend on the chosen image and layer
Saliency	Pixel sensitivity of a class score	Can be noisy; sensitivity is not the same as contribution
Integrated Gradients	Pixel contribution relative to a baseline	Depends on the baseline and requires repeated gradient evaluations
Guided Backpropagation	Fine-grained gradient visualizations	Can look plausible even for an untrained or randomized model
Grad-CAM	Coarse regions supporting a class	Resolution depends on the selected layer
Guided Grad-CAM	Fine detail within a Grad-CAM region	Inherits Guided Backpropagation's caveat
Occlusion	Score change when image patches are hidden	Computationally expensive; results depend on patch size and fill value


Important Notes
- A new nn.Conv2d layer's default initialization is not the same as explicitly applying He initialization in this lab.
- The training comparison is deliberately small, so exact accuracy values can vary with the random seed, software, and hardware. Focus on the general pattern rather than expecting identical numbers.
- Attribution maps are explanations to investigate, not definitive proof of a model's reasoning. Compare methods and use occlusion as a useful reference check.
- If you restart the notebook, rerun the setup and Experiment 1 cells before Experiment 2.
Suggested Experiments
1. Compute fan-in, fan-out, and parameter count for 3×3 and 1×1 convolutions.
2. Compare LeCun and Xavier initialization for nn.Linear(100, 100).
3. Generate Grad-CAM and Guided Backpropagation maps for the second-most-likely class.
4. Add batch normalization after convolution layers and compare activation scales.
5. Change the occlusion window size and observe how the map and runtime change.
Learning Outcomes
After completing the lab, you should be able to:
- Explain fan-in and fan-out and distinguish them from total parameter count.
- Implement and compare LeCun, Xavier, and He initialization.
- Describe how initialization affects activation scales and short-run training behaviour.
- Visualize CNN filters and feature maps.
- Use common gradient-based and perturbation-based attribution methods.
- Explain the purpose and limitations of each visualization method.
References
- PyTorch documentation — torch.nn.init
- Torchvision documentation
- Captum documentation
Based on the accompanying laboratory manual: “Training and Understanding CNNs — Weight Initialization and Visualization with PyTorch.”
