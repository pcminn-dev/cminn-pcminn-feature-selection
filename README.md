# PCMINN

**PCMINN: A GPU-Accelerated Conditional Mutual Information-Based Feature Selection Method**

This repository includes both CPU and GPU implementations of the CMINN algorithm, as well as a real-world case study on bankruptcy prediction.

## 📦 Repository Structure

- `src/` - Core feature selection algorithms (CMINN and PCMINN)
- `case_study/` - Scripts for the bankruptcy dataset experiment (CPU and GPU)
- `requirements.txt` - Required Python packages

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

Make sure you have a compatible CUDA environment for the GPU version.

## 📊 Dataset

This project utilizes the **Company Bankruptcy Prediction** dataset by [fedesoriano](https://www.kaggle.com/fedesoriano) on Kaggle:

> 📂 [Company Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

The dataset comprises financial ratios of Taiwanese companies from 1999 to 2009, with 96 features and a binary bankruptcy label.

### 🔐 Access Instructions

1. **Create a Kaggle account** and log in.
2. Go to [https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) and **accept the terms of use**.
3. Generate an API token:
   - Go to your Kaggle [Account Settings](https://www.kaggle.com/account).
   - Click “Create New API Token” to download `kaggle.json`.
4. Move the token to the appropriate directory:
   - Unix/MacOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
5. Set correct permissions (Unix/MacOS):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
6. Run the dataset download script:
   ```bash
   cd case_study
   python download_bankruptcy_data.py
   ```

## 🧠 Running the Example

```bash
# CPU
python case_study/cminn_bankruptcy.py

# GPU
python case_study/pcminn_bankruptcy.py

## 📄 Citation

If you use this code or any part of the implementation in your work, **please cite the following paper**:

> Papaioannou, N. et al. (2025). *PCMINN: A GPU-Accelerated Conditional Mutual Information-Based Feature Selection Method*.  
> DOI: [https://doi.org/10.xxxx/pcminn.2025](https://doi.org/10.xxxx/pcminn.2025)

## 📜 License

This project is licensed under the MIT License.
