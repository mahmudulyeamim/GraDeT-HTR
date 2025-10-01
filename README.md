# GraDeT-HTR: A Resource-Efficient Bengali Handwritten Text Recognition System utilizing Grapheme-based Tokenizer and Decoder-only Transformer

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2509.18081) [![Website](https://img.shields.io/badge/Website-Demo-blue)](https://cognistorm.ai/hcr) [![Video](https://img.shields.io/badge/Video-YouTube-red)](https://www.youtube.com/watch?v=ckgWBHQarxc) [![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-green)](https://2025.emnlp.org/)

## 📖 Introduction

This is the official GitHub repository for [**GraDeT-HTR: A Resource-Efficient Bengali Handwritten Text Recognition System utilizing Grapheme-based Tokenizer and Decoder-only Transformer**](https://arxiv.org/abs/2509.18081).

**📢 News:** This work has been accepted at the **2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) System Demonstrations**.

**🔗 Resources:**
- 📄 [Paper](https://arxiv.org/abs/2509.18081)
- 🌐 [Live Demo](https://cognistorm.ai/hcr)
- 🎥 [Video Presentation](https://www.youtube.com/watch?v=ckgWBHQarxc)

## 📑 Table of Contents
- [📖 Introduction](#-introduction)
- [⚙️ Installation](#️-installation)
- [🎓 Training](#-training)
- [🚀 Inference](#-inference)
- [📧 Request Access to Weights/Synthetic Datasets](#-request-access-to-weightssynthetic-datasets)
- [📝 Citation](#-citation)

## ⚙️ Installation
Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/mahmudulyeamim/GraDeT-HTR.git
cd GraDeT-HTR
```

### 2. Create a Virtual Environment (Recommended)
It is recommended to create and activate a virtual environment (e.g., venv or Conda) before installing dependencies.

### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Set Up YOLOv5 in BN_DRISHTI
Navigate to the `BN_DRISHTI` directory and clone the YOLOv5 repository:
```bash
cd BN_DRISHTI
git clone https://github.com/ultralytics/yolov5
```

After cloning, replace the default `detect.py` file with the custom version provided in this repository:
```bash
cp detect.py yolov5/detect.py
```

You can now safely delete the `detect.py` file from the `BN_DRISHTI` directory:
```bash
rm detect.py
```

### 5. Download Detection Model Weights
Download the required pre-trained detection model weights into the `model_weights` directory:
```bash
wget -P model_weights https://huggingface.co/crusnic/BN-DRISHTI/resolve/main/models/line_model_best.pt
wget -P model_weights https://huggingface.co/crusnic/BN-DRISHTI/resolve/main/models/word_model_best.pt
```

**Detection model weights provided by:** [BN-DRISHTI](https://github.com/crusnic-corp/BN-DRISHTI)

**Note:** If you don't have `wget` installed, you can install it using your package manager (e.g., `apt-get install wget` on Ubuntu, `brew install wget` on macOS). Alternatively, you can download the files manually from the links above and place them in the `BN_DRISHTI/model_weights` directory.

You're all set! The installation is complete.

## 🎓 Training

### Training the Text Recognition Model
To train the text recognition model, navigate to the `GraDeT_HTR` directory and run the `train.py` file. A sample training dataset is provided to help you test whether the model runs correctly.

**Quick Test with Sample Dataset:**
```bash
cd GraDeT_HTR
python train.py
```

This will train the model using the provided `sample_train` dataset, allowing you to verify that everything is set up properly.

## 🚀 Inference

### Preparing Your Input
Before running inference, create an `input_pages` directory in the root of the project.

```bash
mkdir input_pages
```

Then copy your input images (`.jpg`, `.png`, etc.) or PDF files into this directory. Sample test files are available in the `sample_test` directory for quick testing.

### Running Inference
To perform inference, run the `inference.py` with the path to your pre-trained text recognition model weights.

**For image inputs:**
```bash
python inference.py --weights path/to/your/text_recognition_model_weights.pth
```

**For PDF inputs:** Add the `--pdf true` flag
```bash
python inference.py --weights path/to/your/text_recognition_model_weights.pth --pdf true
```

### Output
The inference results will be saved in the `output_texts` directory (created automatically if it doesn't exist).

## 📧 Request Access to Weights/Synthetic Datasets
To request access to our pre-trained text recognition model weights or synthetic datasets, please email **mahmudulyeamim@gmail.com** with the following information:
- Your name and institutional affiliation
- Intended purpose and use case
- Brief project description
- Confirmation that you will properly acknowledge and cite this work

Access will be granted upon review of your request.

**Important:** Pre-trained weights and synthetic datasets are provided strictly for academic and research purposes.

## 📝 Citation

If you use GraDeT-HTR in your research, please cite our paper:

**ArXiv Preprint:**
```bibtex
@misc{hasan2025gradet,
  title={GraDeT-HTR: A Resource-Efficient Bengali Handwritten Text Recognition System utilizing Grapheme-based Tokenizer and Decoder-only Transformer},
  author={Hasan, Md. Mahmudul and Choudhury, Ahmed Nesar Tahsin and Hasan, Mahmudul and Khan, Md. Mosaddek},
  year={2025},
  eprint={2509.18081},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.18081}
}
```

---

**Contact:** For questions or issues, please open an issue on this repository or contact **mahmudulyeamim@gmail.com**