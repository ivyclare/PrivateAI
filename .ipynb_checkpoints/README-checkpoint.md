# Private-ai

Notebooks for Udacity's [Secure and Private AI course](https://www.udacity.com/course/secure-and-private-ai--ud185).


# Purpose
This repository shows the exercises and projects in the **Secure and Private AI Challenge**. The *deep-learning-v2* folder contains notebooks for an introduction to Neural Networks with Pytorch and the second folder *private-ai* shows us te tools that are necessary to train AI models in a safe and secure fashion. 3 main state of the art privacy preserving methods were covered in the course:
- Federated Learning
- Differential Privacy
- Encrypted Deep Learning

# Projects
## 1. Differential Privacy
    - Project 1: Generate Parallel Databases
 

## Dependencies

To run these notebooks you'll need to install Python 3.6+, PySyft, Numpy, PyTorch 1.0, and Jupyter Notebooks. The easiest way for all of this is to create a conda environment:

```bash
conda create -n pysyft python=3
conda activate pysyft
conda install numpy jupyter notebook
conda install pytorch torchvision -c pytorch
pip install syft
```

