# VeriNews

VeriNews is a Streamlit-based fake news detection app powered by a fine-tuned RoBERTa model.  
It predicts whether a news article is real or fake and helps with verification by surfacing live reference content from the web.

## Model

- Base model: `RoBERTa-base` from Hugging Face Transformers
- Fine-tuned on:
	- LIAR
	- ISOT
	- WELFake

## Features

- Clean web interface built with Streamlit
- RoBERTa-based real/fake classification
- Support for multiple fake-news datasets
- Live reference content retrieval for verification

![App Screenshot](assets/architecture.png)

## Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/yuvrajpinkman/VeriNews.git
cd VeriNews
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Python 3.10+ is required
python -m venv roberta-env

# Windows PowerShell
roberta-env\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model files

Model files are too large for GitHub. Follow the instructions in [Model.md](Model.md).

### 5. Run the app

```bash
streamlit run app.py
```

## Notes

- Ensure model artifacts are placed in the expected `saved_model/` subfolders before launching the app.
- If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## License

This project is licensed under the terms in [LICENSE](LICENSE).