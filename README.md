# TreeReview


This repository contains the implementation code and benchmark dataset for the paper *"TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review"*. It provides a framework for generating comprehensive and in-depth peer reviews using Large Language Models (LLMs) through a hierarchical question-answering process.

## Setup
1. **Create a Conda Environment**:
```shell
conda create -n treereview python=3.11
conda activate treereview
```

2. **Install Dependencies**:
```shell
pip install -r requirements.txt
```

## Usage
To generate a review for a single paper, run the following command:

```shell
python main.py --paper-id <paper_id> --mmd-path <path_to_paper_mmd> --output-path <output_file>
```

