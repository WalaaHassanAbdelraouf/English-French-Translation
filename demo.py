# Demo script to test translation with a trained model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
sys.path.append('.')

from src.inference import translate_text
from config.config import SAVED_MODEL_DIR, MODEL_NAME


def main():
    print("="*60)
    print("Translation Model Demo")
    print("="*60)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    try:
        # Try to load fine-tuned model
        model = AutoModelForSeq2SeqLM.from_pretrained(SAVED_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_DIR)
        print(f"Loaded fine-tuned model from {SAVED_MODEL_DIR}")
    except:
        # Fall back to pretrained model
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loaded pretrained model: {MODEL_NAME}")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    else:
        print("Using CPU")
    
    # Sample texts to translate
    sample_texts = [
        "Hello, how are you?",
        "I love learning new languages.",
        "The weather is beautiful today.",
        "Machine translation has improved significantly.",
        "Thank you for your help!"
    ]
    
    print("\n" + "="*60)
    print("Translation Examples")
    print("="*60)
    
    for i, text in enumerate(sample_texts, 1):
        translation = translate_text(model, tokenizer, text)
        print(f"\nExample {i}:")
        print(f"EN: {text}")
        print(f"FR: {translation}")
    
    print("\n" + "="*60)
    print("Interactive Translation")
    print("="*60)
    print("Enter English text to translate (or 'quit' to exit):")
    
    while True:
        user_input = input("\nEN: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        translation = translate_text(model, tokenizer, user_input)
        print(f"FR: {translation}")


if __name__ == '__main__':
    main()