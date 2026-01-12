# LG-CCR: Cross-Modal Local and Global Alignment for Chinese Character Recognition

Official implementation of the paper **"LG-CCR: Language-Guided Contrastive Contextual Representation for Scene Text Recognition"**.

---

## 📢 News
* **[2026.01.12]** Training code (Pre-training stage) is released.
* **[2025.07.14]** Model architecture and core modules are uploaded.

---

## 🛠️ Installation

We recommend using `conda` to manage your environment:

```bash
# create a new environment
conda create -n lgccr python=3.8 -y
conda activate lgccr

# install dependencies
pip install -r requirements.txt

## 📂 Project Structure
LG-CCR/
├── cfgs/                # Configuration files (.yaml / .py)
├── models/              # Model architecture of LG-CCR
├── datasets/            # Data loading and augmentation pipelines
├── pretrain_main_CCDT.py # Main script for pre-training
└── requirements.txt     # List of dependencies

##🚀 Training
1. Data Preparation

Please organize your datasets in the data/ directory and ensure the paths in your config files are correctly set.

2. Configuration

Before starting the training process, remember to adjust the corresponding parameters in the configuration files located in cfgs/

3.Run Training

Execute the following command to start the pre-training:
python pretrain_main_CCDT.py --cfg cfgs/pretrain_config.py
