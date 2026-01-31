# Training utilities for translation model
from transformers import Trainer, TrainingArguments
import torch


# Setup training arguments for the Trainer.
def setup_training_args(output_dir='results', 
                        num_epochs=3,
                        train_batch_size=4,
                        eval_batch_size=4,
                        learning_rate=3e-5,
                        logging_dir='logs'):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=16,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir=logging_dir,
        logging_steps=500,
        fp16=True if torch.cuda.is_available() else False,
        learning_rate=learning_rate,
        report_to="none",
        gradient_checkpointing=True,
    )
    return training_args


# Train the translation model.
def train_model(model, training_args, tokenized_train, tokenized_val, data_collator, 
                save_dir='models/fine_tuned_marian'):
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    # Train
    trainer.train()
    
    print(f"Training complete! Saving model to {save_dir}")
    # Save model
    trainer.save_model(save_dir)
    
    return trainer