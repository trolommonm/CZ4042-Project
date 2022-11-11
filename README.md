# CZ4042 Project: Gender Classification

<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Project Structure](#installation)
- [Usage](#eyes-usage)
  * [Experiment 1: Levi Hassner](experiment-1-levi-hassner)
  * [Experiment 2: Transfer Learning and Fine-tuning](experiment-2-transfer-learning-and-fine-tuning)
  * [Experiment 3: Supervised Contrastive Learning](experiment-3-supervised-contrastive-learning)
- [Acknowledgements](#acknowledgements)


<!-- About the Project -->
## About the Project

This project aims to build apply several methods using CNN to classify facial
images using the [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html) dataset.

<!-- Getting Started -->
## Getting Started

<!-- Prerequisites -->
### Prerequisites

This project uses Tensorflow and Keras to implement the methods described in the report.

Here are the Python packages required to run the code:
```
tensorflow==2.8.0
tensorflow-addons==0.18.0
pandas==1.4.3
numpy==1.23.4
matplotlib==3.5.2
jupyter==1.0.0
```

Install the dependencies using Pip or Conda package manager.  Note that Tensorflow was installed 
using Pip. Additionally, you need to install cudnn and cudatoolkit for GPU support with Tensorflow. Refer to Tensorflow 
installation guide [here](https://www.tensorflow.org/install/pip).

<!-- Project Structure -->
### Project Structure

Here is the directory layout of our project:
```
.
├── data/               
├── output/                    
├── plots/                     
├── utils/                 
├── dataloader.py                   
├── datapreparation.py
├── finetuning.py
├── levihassner.py
├── pretraining.py
├── supconfinetunining.py
├── supconpretraining.py
└── README.md
```

In order to run the code, please download the datasets and extract them from 
- [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html) 
  - Download the "aligned.tar.gz" files as well as "fold_0_data.txt", "fold_1_data.txt", "fold_2_data.txt", "fold_3_data.txt", "fold_4_data.txt"
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - Download from [aligned and cropped images](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg): "img_align_celeba_png.7z", as well as "list_attr_celeba.txt" and "list_eval_partition.txt"

Ensure that the "data/" folder has the following files and structure:
```
.
├── data/
|   ├── Adience/
|   |   ├── aligned/             # contains all the images extracted from "aligned.tar.gz"
|   |   ├── fold_0_data.txt
|   |   ├── fold_1_data.txt
|   |   ├── fold_2_data.txt
|   |   ├── fold_3_data.txt
|   |   ├── fold_4_data.txt
|   ├── CelebA/
|   |   ├── img_align_celeba/   # contains all the images extracted from "img_align_celeba_png.7z"
|   |   ├── list_attr_celeba.txt
└── ...
```

Run the following Python script to prepare the data for the training process:
```
python datapreparation.py
```

<!-- Usage -->
## Usage

This section will show how to run the various experiments as described in our report.

<!-- Experiment 1 -->
### Experiment 1: Levi Hassner

To run this experiment using the hyperparameters as specified in our report, you can run the following command:

```
python levihassner.py -lr 0.05 -bs 64 --num-epochs 100 -mp
```

<!-- Experiment 2 -->
### Experiment 2: Transfer Learning and Fine-tuning

To run the pre-training stage using the hyperparameters as specified in our report, you can run the following command:
```
python pretraining.py -lr 0.01 -bs 128 --num-epochs 50 -mp
```

To run the fine-tuning stage using the hyperparameters as specified in our report, you can run the following command:
```
python finetuning.py --model-path <path to the pre-trained model on CelebA> -bs 218 --warmup-lr 0.08 --finetune-lr 0.008 --num-epochs 100 -mp
```

<!-- Experiment 3 -->
### Experiment 3: Supervised Contrastive Learning

To run stage 1 using the hyperparameters as specified in our report, you can run the following command:
```
python supconpretraining.py -lr 0.1 -bs 2048 --num-epochs 600 --warmup-epochs 10 --temperature 0.05 -mp
```

To run stage 2 using the optimal learning rate as specified in our report, you can run the following command:
```
python supconfinetuning.py --model-path <path to the pre-model model from stage 1> -bs 128 --num-epochs 100 --num-warmup-epochs 10 -lr 0.05 -mp 
```

The jupyter notebooks in the "plots/" folder contains the code to generate the loss/accuracy graphs and the t-SNE plots in the report.

<!-- Acknowledgments -->
## Acknowledgements

We would like to acknowledge that some parts of our code used in this project were adapted or inspired from other sources, in particular:
- The code for building the ResNet18 architecture was adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
- The code for cosine annealing learning rate scheduler was adapted from https://github.com/Tony607/Keras_Bag_of_Tricks/blob/master/warmup_cosine_decay_scheduler.py
