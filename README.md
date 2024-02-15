# Pix2HTML

## Motivation

Pix2HTML is my implementation of the [Pix2Code](https://arxiv.org/abs/1705.07962) paper. The goal of this project is to convert GUIs created by designers into valid HTML code using an end-to-end deep learning approach. This repository contains all the necessary code to train Pix2HTML using CNNs and LSTMs. You can find the training [notebook on Kaggle](https://www.kaggle.com/code/karimgamaleldin/pix2html) and [model parameters](https://drive.google.com/drive/folders/1Vk6MacZx-Y8MNfdajfcN1kXeTfe0UaUZ).

## Tech Stack

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Architecture

![Diagram](diagram.jpg)

## Results

For text generation in the evaluation phase, I used:

1. Greedy Search
2. Beam Search (Beam = 3)
3. Beam Search (Beam = 5)

and the BLEU scores were as follows:

| Model       | Greedy search | Beam Search, Beam = 3 | Beam Search, Beam = 5 |
| ----------- | ------------- | --------------------- | --------------------- |
| VGG + LSTMs | ~0.78         | ~0.80                 | ~0.80                 |

More experiments are coming.

## Credits

- [Tony Beltramelli](https://github.com/tonybeltramelli)
- [Pix2Code Paper](https://arxiv.org/abs/1705.07962)
