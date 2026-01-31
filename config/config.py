# Model Configuration
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-fr'

# Data Configuration
DATA_PATH = 'data/en-fr.csv'
NUM_SAMPLES = 100000  # Number of samples to load from dataset

# Text Preprocessing
MIN_LENGTH = 1  # Minimum number of words
MAX_LENGTH = 128  # Maximum number of words
MAX_TOKEN_LENGTH = 512  # Maximum token length for model

# Data Split Ratios
TEST_SIZE = 0.15
VAL_SIZE = 0.175  
RANDOM_STATE = 42

# Training Configuration
OUTPUT_DIR = 'results'
LOGGING_DIR = 'logs'
SAVED_MODEL_DIR = 'models/fine_tuned_model'

# Training Arguments
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 3e-5
LOGGING_STEPS = 500

# Evaluation Configuration
NUM_BEAMS = 4  
EARLY_STOPPING = True

# Demo Configuration
NUM_TRANSLATION_EXAMPLES = 5