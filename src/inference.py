import torch

# Display translation examples from the test dataset.
def show_translation_examples(model, tokenizer, test_dataset, num_examples=5):
    model.eval()
    
    print("\nTranslation Examples:")
    print("=" * 60)
    
    for i in range(min(num_examples, len(test_dataset))):
        ex = test_dataset[i]
        
        # Prepare input
        inp = tokenizer(ex['en'], return_tensors='pt', truncation=True).to(model.device)
        
        # Generate translation
        with torch.no_grad():
            out = model.generate(**inp, max_length=512, num_beams=4)
        
        # Decode prediction
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        
        print(f"\nExample {i+1}:")
        print(f"{'='*60}")
        print(f"EN:        {ex['en']}")
        print(f"REF (FR):  {ex['fr']}")
        print(f"PRED (FR): {pred}")
    
    print(f"\n{'='*60}")
    print(f"Displayed {min(num_examples, len(test_dataset))} translation examples")


def translate_text(model, tokenizer, text):
    model.eval()
    
    # Prepare input
    inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    
    # Generate translation
    with torch.no_grad():
        out = model.generate(**inp, max_length=512, num_beams=4, early_stopping=True)
    
    # Decode and return
    translation = tokenizer.decode(out[0], skip_special_tokens=True)
    
    return translation