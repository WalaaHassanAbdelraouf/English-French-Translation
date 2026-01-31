# Main pipeline
from transformers import AutoTokenizer
import sys
sys.path.append('.')

from src.data_loader import load_data
from src.data_cleaning import explore_data, data_cleaning, text_cleaning
from src.data_preprocessing import data_splitting, tokenize_datasets
from src.model_setup import setup_model_and_collator
from src.train import setup_training_args, train_model
from src.evaluate import evaluate_model
from src.inference import show_translation_examples
from config.config import *


def main():
    print("="*60)
    print("Translation Model Training Pipeline")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_data(DATA_PATH, nrows=NUM_SAMPLES)
    print(f"Loaded {len(df)} rows")
    print(df.head())
    
    # Explore data
    print("\nExploring data...")
    explore_data(df)
    
    # Clean data
    print("\nCleaning data...")
    df = data_cleaning(df)
    
    # Normalize text
    print("\nNormalizing text...")
    df = text_cleaning(df)
    
    # Split data
    print("\nSplitting data...")
    train_dataset, val_dataset, test_dataset = data_splitting(df)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_train, tokenized_val, tokenized_test = tokenize_datasets(
        train_dataset, val_dataset, test_dataset, tokenizer
    )
    
    # Setup model and collator
    print("\nSetting up model and data collator...")
    model, data_collator = setup_model_and_collator(tokenizer, MODEL_NAME)
    
    # Setup training and train
    print("\nTraining model...")
    training_args = setup_training_args(
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=LOGGING_DIR
    )
    
    trainer = train_model(
        model=model,
        training_args=training_args,
        tokenized_train=tokenized_train,
        tokenized_val=tokenized_val,
        data_collator=data_collator,
        save_dir=SAVED_MODEL_DIR
    )
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    score, predictions, references = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset
    )
    
    # Show examples
    print("\n" + "="*60)
    print("Translation Examples")
    print("="*60)
    show_translation_examples(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_examples=NUM_TRANSLATION_EXAMPLES
    )
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Final BLEU Score: {score.score:.2f}")
    print(f"Model saved to: {SAVED_MODEL_DIR}")


if __name__ == '__main__':
    main()