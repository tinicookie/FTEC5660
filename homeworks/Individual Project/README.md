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
.
├── Modification-Task3.ipynb              # Main notebook: data loading, inference, evaluation
├── README.md                 # This file
```

## Installation

1. **Clone the repository** (or download the notebook).

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install langchain-google-genai langchain-core scikit-learn imbalanced-learn tqdm jupyter
   ```
   (A `requirements.txt` is provided below – you can create it from the list.)

4. **Obtain a Google AI Studio API key**  
   - Go to [Google AI Studio](https://aistudio.google.com/) and create an API key.  
   - In Google Colab, you can store it as a secret named `VERTEX_API_KEY`.  
   - If running locally, set the environment variable `VERTEX_API_KEY` or replace the key in the notebook (not recommended for sharing).

5. **Download the UNBench dataset**  
   The full dataset is available from the original repository:  
   [https://drive.google.com/file/d/1tiBCCYPjeIN92TkO8Vt8vrpSKLmGb-6Y/view?usp=sharing](https://drive.google.com/file/d/1tiBCCYPjeIN92TkO8Vt8vrpSKLmGb-6Y/view?usp=sharing)  
   Download and unzip it. Ensure the file `task3_test.json` is placed at `/content/UNBench-all/task3_test.json` (the notebook expects this path; modify if necessary).

## Usage

1. **Open the notebook**  
   ```bash
   jupyter notebook Modification-Task3.ipynb
   ```

2. **Run all cells** sequentially.  
   - The first cell installs required packages (if not already installed).  
   - The second cell downloads the dataset (you may skip this if you have already downloaded it manually).  
   - The third cell loads the test data.  
   - The fourth cell initializes the Gemini model.  
   - The fifth cell runs inference on all 989 test samples (may take ~1 hour on a Colab GPU).  
   - The final cells compute and print the evaluation metrics.

3. **Expected output**  
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
