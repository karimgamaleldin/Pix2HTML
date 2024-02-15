# Pix2HTML

## Motivation

Pix2HTML is my implementation for [Pix2Code](https://arxiv.org/abs/1705.07962) paper. The aim of this project is to convert GUIs created by a designer into valid HTML by taking an end-to-end deep learning approach. In this repo, you will find all the code you need to train Pix2HTML using CNNs and LSTMs. You can find the [notebook](https://www.kaggle.com/code/karimgamaleldin/pix2html) kaggle.

## Tech Stack

<div align="center>
  
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
  
</div>

## Architecture

![Diagram](diagram.jpg)

## Results

For text generation in the evaluation phase I used

1. Greedy Search
2. Beam Search (Beam = 3)
3. Beam Search (Beam = 5)

and the BLEU score was as follows:

| Modek       | Greedy search | Beam Search, Beam = 3 | Beam Search, Beam = 5 |
| ----------- | ------------- | --------------------- | --------------------- |
| VGG + LSTMs | ~ 0.78        | ~80                   | ~80                   |

more experiments are coming

## Credits

- [Tony Beltramelli][https://github.com/tonybeltramelli]
- [Pix2Code](https://arxiv.org/abs/1705.07962)
