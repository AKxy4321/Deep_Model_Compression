---

# Deep Model Compression using HBFP

This repository contains code for compressing deep learning models using History Based Filter Pruning (HBFP). The project includes implementations for LeNet-5 on the MNIST dataset, ResNet-56 on the CIFAR-10 dataset, and VGG-16 on the CIFAR-10 dataset. Various pruning methods are used to optimize the model by removing less important filters.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Pruning Methods](#pruning-methods)
  - [L1 Norm](#l1-norm)
  - [Cosine Similarity](#cosine-similarity)
  - [Pearson Coefficient](#pearson-coefficient)
- [Results](#results)
- [Ackwnoledgements](#ackwnoledgements)

## Overview

This project implements History Based Filter Pruning (HBFP) to compress deep learning models. The models used include LeNet-5 for the MNIST dataset, ResNet-56 on the CIFAR-10 dataset, and VGG-16 for the CIFAR-10 dataset. HBFP is applied using three different methods: L1 norm, cosine similarity, and Pearson coefficient.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AKxy4321/Deep_Model_Compression
    cd Deep_Model_Compression
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train and prune a model, follow these steps:

1. Prepare the dataset (MNIST or CIFAR-10).
2. Train the model using the provided training scripts.
3. Apply HBFP using one of the pruning methods.
4. Evaluate the pruned model's performance.

## Models
- **LeNet5** on the MNIST dataset
- **ResNet56** on the CIFAR-10 dataset
- **VGG16** on the CIFAR-10 dataset

## Pruning Methods

### L1 Norm

Prunes filters based on their L1 norm. Filters with the smallest L1 norm are considered less important and are pruned first.

### Cosine Similarity

Prunes filters based on the cosine similarity between them. Filters that are very similar to each other are redundant, and the less important ones are pruned.

### Pearson Coefficient

Prunes filters based on the Pearson correlation coefficient. Filters that are highly correlated with each other are redundant, and the less important ones are pruned.

## Results
Detailed results of the pruning experiments can be found in the [results](results/) directory.

## Ackwnoledgements

<b>Original work:</b> S.H. Shabbeer Basha, Mohammad Farazuddin, Viswanath Pulabaigari, Shiv Ram Dubey, Snehasis Mukherjee
[GitHub Repository](https://github.com/shabbeersh/History_Based_Filter_Pruning)

---
