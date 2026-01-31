# English to French Machine Translation

An implementation of transformer-based English-to-French machine translation using the Helsinki-NLP OPUS-MT model.

## Project Overview

In this project, I explored transformer-based models for machine translation. The task focuses on English-to-French translation using a large-scale parallel dataset.

The original dataset contains around 20 million sentence pairs, which makes it very rich in linguistic context and phrasing. However, since this is my first practical experiment with transformer-based translation models, and due to computational and resource limitations, a subset of 100,000 samples was randomly selected and used for this project.

**The main goal of this project is to:**
- Understand the full workflow of transformer-based translation.
- Experiment with fine-tuning a pre-trained model.
- Evaluate translation quality using standard metrics.

## Dataset

- **Source**: [en-fr-translation-dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset)
- **Languages**: English (source) → French (target)
- **Original size**: Over 22.5 million samples
- **Default subset**: 100,000 samples 

## Project Structure

```
ENGLISH-FRENCH-TRANSLATION/
├── config/
│   └── config.py                 # Configuration settings
├── src/
│   ├── data_loader.py           # Data loading utilities
│   ├── data_cleaning.py         # Data cleaning functions
│   ├── data_preprocessing.py    # Preprocessing and tokenization
│   ├── model_setup.py           # Model initialization
│   ├── train.py                 # Training utilities
│   ├── evaluate.py              # Evaluation metrics
│   └── inference.py             # Translation inference
├── data/                         # Dataset directory
├── models/                       # Saved models directory
├── outputs/                      # Output directory
├── main.py                       # Main training pipeline
├── demo.py                       # Interactive demo script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/WalaaHassanAbdelraouf/English-to-French-Translator.git
cd English-to-French-Translator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Place your `en-fr.csv` file in the `data/` directory
   - Or update the `DATA_PATH` in `config/config.py`

## Usage

### Training the Model

Run the complete training pipeline:

```bash
python main.py
```

This will:
1. Load and explore the data
2. Clean and preprocess the text
3. Split into train/val/test sets
4. Tokenize the datasets
5. Fine-tune the model
6. Evaluate on the test set
7. Display sample translations

### Running the Demo

Test the trained model interactively:

```bash
python demo.py
```

The demo provides:
- Pre-defined translation examples
- Interactive mode for custom translations

## Configuration

All hyperparameters and settings can be modified in `config/config.py`:

```python
# Model Configuration
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-fr'

# Training Configuration
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
LEARNING_RATE = 3e-5

# Data Configuration
NUM_SAMPLES = 100000
MIN_LENGTH = 1
MAX_LENGTH = 128
```

## Training Details

- **Model**: Helsinki-NLP/opus-mt-en-fr (MarianMT)
- **Training Epochs**: 3
- **Batch Size**: 4 (with gradient accumulation)
- **Learning Rate**: 3e-5
- **Optimizer**: AdamW (via Hugging Face Trainer)
- **Evaluation Metric**: BLEU Score

## Results

- **BLEU Score**: ~58.69% on test set
- The model achieves strong translation quality given the training constraints

## Pipeline Workflow

1. **Data Loading**: Load translation pairs from CSV
2. **Data Cleaning**: 
   - Remove missing values
   - Filter by sentence length (1-128 words)
   - Text normalization (lowercase, strip)
3. **Preprocessing**:
   - Split data (70% train, 15% val, 15% test)
   - Tokenization with model-specific tokenizer
   - Data collation with padding
4. **Training**:
   - Fine-tune pre-trained model
   - Mixed precision training (FP16 on GPU)
   - Gradient checkpointing for memory efficiency
5. **Evaluation**:
   - BLEU score computation
   - Sample translation visualization

## Dependencies

- `transformers`: Hugging Face transformers library
- `torch`: PyTorch deep learning framework
- `datasets`: Hugging Face datasets library
- `pandas`: Data manipulation
- `scikit-learn`: Data splitting
- `sacrebleu`: BLEU score computation
- `tqdm`: Progress bars

## License

This project is licensed under the MIT License.

