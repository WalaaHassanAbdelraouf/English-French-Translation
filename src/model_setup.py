# Model setup utilities 
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch
import os

# Setup the translation model and data collator.
def setup_model_and_collator(tokenizer, model_name='Helsinki-NLP/opus-mt-en-fr'):
    # Disable wandb if causing issues
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    else:
        print("Using CPU")
        
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    print("Model and data collator ready!")
    
    return model, data_collator