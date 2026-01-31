# Model evaluation utilities
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm

# Evaluate the model on test dataset using BLEU score.
def evaluate_model(model, tokenizer, test_dataset):
    # Set model to evaluation mode
    model.eval()
    
    # Generate predictions
    predictions = []
    references = []
    
    print(f"Generating predictions for entire test set ({len(test_dataset)} samples)...")
    
    with torch.no_grad():  # Disable gradient calculations for speed
        for example in tqdm(test_dataset, desc="Evaluating"):
            inputs = tokenizer(
                example['en'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Generate translation
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,  
                early_stopping=True
            )
            
            # Decode prediction
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred)
            references.append([example['fr']])  # sacrebleu expects list of references
    
    # Compute BLEU score
    bleu = BLEU()
    score = bleu.corpus_score(predictions, references)
    
    print(f"\nBLEU Score: {score.score:.2f}")
    print(f"Evaluated {len(predictions)} translations")
    
    return score, predictions, references