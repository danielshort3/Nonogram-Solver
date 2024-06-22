# Nonogram Solver

This repository contains a project aimed at solving Nonogram puzzles using a reinforcement learning approach. The project uses a deep learning model to predict the next moves in the Nonogram puzzle and includes helper functions for generating puzzles, calculating clues, and updating puzzle states.

## Table of Contents

- [Introduction](#introduction)
- [Pre-trained Model](#pre-trained-model)
- [Solving Puzzles](#solving-puzzles)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This project aims to solve Nonogram puzzles using a reinforcement learning approach. The process involves generating unique Nonogram puzzles, training a model to solve them, and evaluating the model's performance. The project includes various helper functions for generating puzzles, calculating clues, saving and loading checkpoints, and more.

## Pre-trained Model

The repository includes the following pre-trained model:
- A model trained to solve Nonogram puzzles using a combination of convolutional neural networks (CNN) and transformers for handling clues.

This models is provided as a state dictionary and can be found in the `models/` directory.

## Solving Puzzles

The Nonogram agent is trained using the Policy Gradient method. It selects actions based on the current state and clues, and the environment updates the puzzle state accordingly. The agent is rewarded based on the correctness of its guesses.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/danielshort3/nonogram.git
    cd nonogram
    ```

2. Install the required packages (make sure you have `pip` and `virtualenv` installed):
    ```bash
    pip install torch torchvision
    pip install tensorboard
    ```

3. Ensure the pre-trained model weights are in the `models/` directory.

## Usage

1. Run the code from beginning to end to train the model using `Nonogram.ipynb` notebook.

2. To test the pretrained model, run the code skipping the `Main Execution` section and instead running the `Testing Main Execution` section.
