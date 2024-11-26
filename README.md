Certainly! Here's a sample **README** for your GitHub project, based on the details you've provided:

---

# TDG-Mamba: Advanced Spatiotemporal Embedding for Temporal Dynamic Graph Learning via Bidirectional Information Propagation

## Overview

**TDG-Mamba** is an advanced spatiotemporal embedding model designed for temporal dynamic graph learning. The model incorporates **bidirectional information propagation** to enhance the learning process and improve the performance of graph-based models on dynamic and temporal data.

This repository contains the code implementation of the model, as presented in the paper titled "**TDG-Mamba: Advanced Spatiotemporal Embedding for Temporal Dynamic Graph Learning via Bidirectional Information Propagation**", which was published in the *IEEE Transactions on Computational Social Systems (TCSS)*.


## Installation

To use this repository, you need to have Python 3.7+ installed, along with some essential libraries. You can install the necessary dependencies using `pip`.

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/TDG-Mamba.git
   cd code
   ```

2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

For dataset-related instructions, please refer to the specific datafolder in this repository.

### Quick Start

To train the TDG-Mamba model, run the following command:

```bash
python main.py --data [facebook/game_of_thrones/formula/enron]
```

This will start training the model using the configuration specified in `config_file.yaml`. Make sure to adjust the parameters in the configuration file according to your dataset and computational environment.


## Citation

If you use this code or model in your research, please cite the following paper:

```
@article{your_paper_reference,
  author = {Mengran Li, Junzhou Chen, Bo Li, Yong Zhang, Ronghui Zhang, Siyuan Gong, Xiaolei Ma, and Zhihong Tian},
  title = {TDG-Mamba: Advanced Spatiotemporal Embedding for Temporal Dynamic Graph Learning via Bidirectional Information Propagation},
  journal = {IEEE Transactions on Computational Social Systems},
  year = {2024}
}
```

## Contributing

We welcome contributions to this project! If you find bugs or would like to add new features, feel free to fork the repository and submit a pull request.


## Acknowledgements

- We would like to thank the authors of previous works in temporal dynamic graph learning and spatiotemporal embedding models, whose contributions inspired the development of this model.

## Contact

For questions or inquiries, please contact [limr39@mail2.sysu.edu.cn].



