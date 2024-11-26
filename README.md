Certainly! Here's a sample **README** for your GitHub project, based on the details you've provided:

---

# TDG-Mamba: Advanced Spatiotemporal Embedding for Temporal Dynamic Graph Learning via Bidirectional Information Propagation

## Overview

**TDG-Mamba** is an advanced spatiotemporal embedding model designed for temporal dynamic graph learning. The model incorporates **bidirectional information propagation** to enhance the learning process and improve the performance of graph-based models on dynamic and temporal data.

This repository contains the code implementation of the model, as presented in the paper titled "**TDG-Mamba: Advanced Spatiotemporal Embedding for Temporal Dynamic Graph Learning via Bidirectional Information Propagation**", which was published in the *IEEE Transactions on Computational Social Systems (TCSS)*.


## Installation

To use this repository, you need to have Python 3.8+ installed, along with some essential libraries. You can install the necessary dependencies using `pip`.

1. Clone this repository:

   ```bash
   git clone https://github.com/limengran98/TDG-Mamba.git
   cd code
   ```

2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

This repository includes several datasets for training and evaluation. Specifically, the **Facebook dataset** is provided for use with the TDG-Mamba model. For detailed instructions on how to access and use the datasets, please refer to the [data folder in the GraphERT repository](https://github.com/moranbel/GraphERT/tree/main/data), which includes additional datasets and documentation.

### Quick Start

To train the TDG-Mamba model, run the following command:

```bash
python main.py --data [facebook/game_of_thrones/formula/enron]
```

Make sure to adjust the appropriate parameters in the configuration according to your dataset and computational environment.


## Citation

If you use this code or model in your research, please cite the following paper:

```
@article{li2024mamba,
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



