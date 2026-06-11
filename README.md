# Implementation of Meta-Learning Algorithm: Reptile

This repository contains an implementation and empirical evaluation of **Reptile**, a first-order meta-learning algorithm. The goal of this project is to study how a model can learn a good initialization that adapts quickly to new tasks using only a small amount of data.

This project was developed as part of the **Continual Learning** course at the **University of Pisa**.

---

## Project Overview

Meta-learning, also known as **learning to learn**, focuses on training models that can adapt rapidly to new tasks. Instead of learning one fixed task, the model is trained across many small tasks so that it can quickly learn unseen tasks with only a few examples.

This project implements Reptile and evaluates it on:

- Sine wave regression
- Omniglot 5-way 1-shot classification
- Omniglot 5-way 5-shot classification
- Omniglot 20-way 5-shot classification
- Omniglot 20-way 1-shot classification

---

## Reptile Algorithm

Reptile uses two optimization levels: an **inner loop** and an **outer loop**.

In the **inner loop**, the current meta-model is copied and adapted to one sampled task using the support set.

In the **outer loop**, the original meta-model is moved toward the adapted task model.

The main Reptile update is:

```text
meta_model ← meta_model + ε(adapted_model − meta_model)
```

This update teaches the model to learn an initialization that can quickly adapt to future tasks.

---

## Sine Wave Regression

The sine wave regression experiment was used as a proof-of-concept before moving to image classification.

Each task is a different sine curve, generated using a random amplitude and phase:

```text
y = A sin(x + phi)
```

The model is trained to adapt quickly to a new sine wave from a small support set.

### Model

```text
MLP: 1 → 64 → 64 → 1
```

### Metric

Since this is a regression task, accuracy is not used. Performance is measured using **Mean Squared Error (MSE)**.

### Result

The MSE decreased from approximately **3.6** to around **0.5–0.6**, showing that the model learned to adapt better to new sine wave tasks.

---

## Omniglot Few-Shot Classification

Omniglot is a dataset of handwritten characters from many alphabets. In this project, each character is treated as a separate class.

A few-shot task is created by sampling:

- `n_way` character classes
- `k_shot` support images per class
- `q_query` query images per class

The support set is used for inner-loop adaptation, and the query set is used for evaluation.

---

## Dataset Splitting Strategy

The Omniglot `background` and `evaluation` folders were merged into one larger character pool. After merging, the dataset was split at the character-class level.

This means that meta-training and meta-testing use disjoint character classes. Therefore, even though the folders were merged before splitting, there is no class leakage between training and testing tasks.

---

## CNN Architecture for Omniglot

The Omniglot experiments use a compact convolutional neural network.

Each convolutional block contains:

```text
Conv2d → BatchNorm2d → ReLU
```

The network uses four stride-2 convolutional blocks. The input image has size `1 × 28 × 28`, and after four blocks it becomes a compact `64 × 2 × 2` feature map. This is flattened into a 256-dimensional vector and passed to a classifier.

```text
Input: 1 × 28 × 28
↓
ConvBlock(1, 64)
↓
ConvBlock(64, 64)
↓
ConvBlock(64, 64)
↓
ConvBlock(64, 64)
↓
Flatten: 64 × 2 × 2 = 256
↓
Linear(256, n_way)
↓
LogSoftmax
```

`LogSoftmax` is used together with `NLLLoss`.

---

## Final Results

| Experiment | Setting | Result |
|---|---|---|
| Sine Wave Regression | MLP regression | MSE reduced from ~3.6 to ~0.5–0.6 |
| Omniglot 5-way 1-shot | 5 classes, 1 support image/class | 93.56% test accuracy |
| Omniglot 5-way 5-shot | 5 classes, 5 support images/class | 97.23% test accuracy |
| Omniglot 20-way 5-shot | 20 classes, 5 support images/class | 88.39% test accuracy |
| Omniglot 20-way 1-shot | 20 classes, 1 support image/class | 85.27% test accuracy |

---

## Implementation Details

The implementation includes:

- Episodic few-shot task sampling
- Inner-loop task adaptation
- Outer-loop Reptile meta-update
- BatchNorm-based CNN architecture
- Parameter-only meta-updates
- Checkpointing for long training runs
- Validation on held-out character classes
- Support/query evaluation protocol

---

## Key Challenges

Several implementation details strongly affected performance.

The most important challenge was correctly interpreting `inner_steps`. In Reptile, one inner step should mean one mini-batch optimizer update, not one full epoch over the support set. Treating inner steps incorrectly caused unstable validation behavior and support-set overfitting.

Another challenge was meta-learning-rate scheduling. Fixed meta-learning rates either plateaued too early or destabilized training. Linear meta-learning-rate annealing improved stability.

BatchNorm also required careful handling. The outer Reptile update was applied only to learnable parameters, avoiding direct corruption of BatchNorm running statistics such as `running_mean` and `running_var`.

---

## Repository Structure

```text
.
├──  Reptile Project-sine_wave/
│   ├── data.py
│   ├── model.py
│   └── train.py
│
├── Research Papers/
│   ├── On First-Order Meta-Learning Algorithms
│   ├── Reptile___a_Scalable_Metalearning
│  
├──Presentation/ 
│   ├── Project Proposal_Continual Learninng.pdf
│   ├── Reptile Meta-Learning Presentation.pdf
├── notebooks/
│   ├── final-model-5-way-1-shot.ipynb
│   ├── rep-5-way-5-best-model.ipynb
│   ├── 20-way-5-shot-final-model.ipynb
│   └── omniglot-20-way-1-shot.ipynb
│
├── plots/
│   ├── adaptation_plot.png
│   └── mse_curve.png
│
└── README.md
```

The exact file names may differ depending on the local version of the project.

---

## How to Run

Install the required packages:

```bash
pip install torch torchvision numpy matplotlib pillow
```

Run the sine wave experiment:

```bash
python train.py
```

For Omniglot experiments, open the corresponding notebook and run the cells in order.

Recommended notebook execution order:

```text
1. Load dataset
2. Build class dictionary
3. Create train/validation/test splits
4. Define task sampler
5. Define CNN model
6. Run Reptile training
7. Evaluate final model
8. Plot results
```

---

## Conclusion

The experiments show that Reptile can learn a useful initialization for fast adaptation. The method performs strongly on Omniglot few-shot classification, especially in the 5-way settings.

The project also shows that Reptile is simple conceptually but sensitive in practice. Correct inner-loop implementation, meta-learning-rate scheduling, BatchNorm handling, and sufficient training time are important for stable performance.

---

## Future Work

Possible extensions include:

- Running multiple random seeds and reporting confidence intervals
- Performing wider hyperparameter sweeps
- Extending the 20-way experiments with longer training
- Evaluating on harder datasets such as Mini-ImageNet

---

## Author

**Shaikh Asif Hossain**  
**Sorana Resiga**
University of Pisa  
Continual Learning Project
