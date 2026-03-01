根据你提供的GitHub仓库路径和文件名，我已经为你修改了README文档，使其指向你实际的代码位置。以下是修正后的版本，你可以直接复制使用。

---

# UNBench Task 3 Reproducibility with Gemini

This repository contains a reproducibility experiment for **Task 3 (Draft Adoption Prediction)** of the [UNBench](https://github.com/yueqingliang1/UNBench) project. The original implementation used models via the TogetherAI API; here the model is replaced with Google's **gemini-2.5-flash** (accessed through LangChain) while keeping all other components (data, prompts, evaluation) identical.

## Paper Reference

**Benchmarking LLMs for Political Science: A United Nations Perspective**  
*Yueqing Liang, Liangwei Yang, Chen Wang, Congying Xia, Rui Meng, Xiongxiao Xu, Haoran Wang, Ali Payani, Kai Shu*  
**AAAI 2026 (Oral)**  
📄 [arXiv:2502.14122](https://arxiv.org/abs/2502.14122)

## Overview

UNBench is a multi‑stage benchmark built on United Nations Security Council (UNSC) records. It evaluates LLMs on four tasks spanning drafting, voting, and statement generation.  
**Task 3** (Draft Adoption Prediction) is a binary classification problem: given the full text of a draft resolution, predict whether it will be adopted (`1`) or not (`0`).

In this fork, the original TogetherAI backend is replaced with `gemini-2.5-flash` via `langchain-google-genai`. The test set (989 samples, as described in the paper) is used without modification, and the same evaluation metrics are computed.

## What Has Been Changed

| Original (TogetherAI)          | This Repository (Gemini)               |
|--------------------------------|----------------------------------------|
| `from together import Together` | `from langchain_google_genai import ChatGoogleGenerativeAI` |
| `client.chat.completions.create` | `llm.invoke(messages)`                |
| API key: TogetherAI             | API key: Google AI Studio (`VERTEX_API_KEY`) |
| Model: e.g., `meta-llama/Llama-3-70b-chat-hf` | Model: `gemini-2.5-flash` |

All prompts and evaluation scripts remain exactly as in the original UNBench repository.

## Project Structure

```
FTEC5660/
└── homeworks/
    └── Individual Project/
        ├── Modification-Task 3.ipynb   # Main notebook: data loading, inference, evaluation
        └── README.md                    # This file
```

## Getting Started

### 1. Obtain the Notebook
- Navigate to the repository: [https://github.com/tinicookie/FTEC5660/tree/main/homeworks/Individual%20Project](https://github.com/tinicookie/FTEC5660/tree/main/homeworks/Individual%20Project)
- Download or clone the entire repository, or directly download the `Modification-Task 3.ipynb` file.

### 2. Set Up Environment

#### Option A: Run Locally with Jupyter
- Ensure you have Python 3.9+ installed.
- (Optional) Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate   # On Windows: venv\Scripts\activate
  ```
- Install required packages:
  ```bash
  pip install langchain-google-genai langchain-core scikit-learn imbalanced-learn tqdm jupyter
  ```
- Start Jupyter:
  ```bash
  jupyter notebook
  ```
- Navigate to the `homeworks/Individual Project/` folder and open `Modification-Task 3.ipynb`.

#### Option B: Run on Google Colab
- Go to [Google Colab](https://colab.research.google.com/).
- Select **File** → **Upload notebook** and upload `Modification-Task 3.ipynb`.
- Make sure you have a Google AI Studio API key and store it in Colab's secrets as `VERTEX_API_KEY` (🔑 Secrets tab).

### 3. Download the Dataset
The full dataset is available from the original UNBench repository:  
[https://drive.google.com/file/d/1tiBCCYPjeIN92TkO8Vt8vrpSKLmGb-6Y/view?usp=sharing](https://drive.google.com/file/d/1tiBCCYPjeIN92TkO8Vt8vrpSKLmGb-6Y/view?usp=sharing)  
Download and unzip it. The notebook expects the file `task3_test.json` to be located at `/content/UNBench-all/task3_test.json` (default Colab path after running the download cell). If you place it elsewhere, update the path in the third code cell.

### 4. Run the Notebook
Execute all cells sequentially.  
- The first cell installs required packages.  
- The second cell downloads and extracts the dataset (you may skip this if you have already done so manually).  
- The third cell loads the test data.  
- The fourth cell initializes the Gemini model.  
- The fifth cell runs inference on all 989 test samples (may take ~1 hour on a Colab GPU).  
- The final cells compute and print the evaluation metrics.

## Expected Output

After successful execution, you should see metrics similar to:
```
Accuracy: 0.9696663296258847
AUC: 0.9305757135945815
Balanced Accuracy: 0.9305757135945816
Precision: 0.6619718309859155
Recall: 0.8867924528301887
F1: 0.7580645161290323
PR AUC: 0.7774155089454636
MCC: 0.7512852629407566
G-Mean: 0.9295451494192747
Specificity: 0.9743589743589743
```

*(These numbers were obtained with `gemini-2.5-flash` on the full test set.)*

## Requirements

- Python 3.9+
- `langchain-google-genai`
- `langchain-core`
- `scikit-learn`
- `imbalanced-learn`
- `tqdm`
- `jupyter`
- `gdown` (optional, for automatic dataset download)

You can install all with:
```bash
pip install langchain-google-genai langchain-core scikit-learn imbalanced-learn tqdm jupyter gdown
```

## Citation

If you use this code or the UNBench dataset in your research, please cite the original paper:

```bibtex
@inproceedings{liang2026unbench,
  title={Benchmarking LLMs for Political Science: A United Nations Perspective},
  author={Liang, Yueqing and Yang, Liangwei and Wang, Chen and Xia, Congying and Meng, Rui and Xu, Xiongxiao and Wang, Haoran and Payani, Ali and Shu, Kai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

This project follows the original UNBench license (MIT). See the [LICENSE](LICENSE) file for details (if included).