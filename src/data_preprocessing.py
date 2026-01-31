# Data preprocessing utilities 
from sklearn.model_selection import train_test_split
from datasets import Dataset


def data_splitting(df):
    temp_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(temp_df, test_size=0.175, random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, val_dataset, test_dataset


# Tokenize input and target texts.
def preprocessing(examples, tokenizer):
    inputs = examples['en']
    targets = examples['fr']
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        padding='longest',
        truncation=True
    )
    
    labels = tokenizer(
        targets, 
        max_length=512,
        padding='longest',
        truncation=True
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# Apply tokenization to all dataset splits.
def tokenize_datasets(train_dataset, val_dataset, test_dataset, tokenizer):
    tokenized_train = train_dataset.map(
        lambda x: preprocessing(x, tokenizer), 
        batched=True, 
        batch_size=1000
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocessing(x, tokenizer), 
        batched=True, 
        batch_size=1000
    )
    tokenized_test = test_dataset.map(
        lambda x: preprocessing(x, tokenizer), 
        batched=True, 
        batch_size=1000
    )
    
    print(f"Tokenized train: {len(tokenized_train)} samples")
    print(f"Tokenized val: {len(tokenized_val)} samples")
    print(f"Tokenized test: {len(tokenized_test)} samples")
    
    return tokenized_train, tokenized_val, tokenized_test