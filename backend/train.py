import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_metric
from model import T5Summarizer
from dataset_loader import CNNDailyMailDataLoader

class ModelTrainer:
    def __init__(self, model: T5Summarizer):
        self.model = model
        self.dataset_loader = CNNDailyMailDataLoader()
        
        # Initialize ROUGE metric for evaluation
        try:
            self.rouge = load_metric('rouge')
            logging.info("ROUGE metric loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load ROUGE metric: {str(e)}. Evaluation will be limited to loss.")
            self.rouge = None

    def train(self, num_epochs: int = 3, batch_size: int = 8,
              eval_steps: int = 100, save_steps: int = 500,
              checkpoint_dir: Optional[str] = None,
              learning_rate: float = 1e-4,
              warmup_steps: int = 500) -> Dict[str, float]:
        """Train the model using CNN/DailyMail dataset.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            eval_steps: Steps between evaluations
            save_steps: Steps between model checkpoints
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Load training and validation data
            logging.info("Loading and preprocessing training data...")
            train_data = self.dataset_loader.load_dataset('train')
            logging.info("Loading and preprocessing validation data...")
            eval_data = self.dataset_loader.load_dataset('validation')
            
            if not train_data or not eval_data:
                raise ValueError("Failed to load dataset")
            
            logging.info(f"Loaded {len(train_data)} training examples and {len(eval_data)} validation examples")
                
            # Prepare datasets with enhanced preprocessing
            logging.info("Preparing training dataset...")
            train_dataset = self.model._prepare_dataset(train_data, is_cnn_dailymail=True)
            logging.info("Preparing validation dataset...")
            eval_dataset = self.model._prepare_dataset(eval_data, is_cnn_dailymail=True)
            
            if not train_dataset or not eval_dataset:
                raise ValueError("Failed to prepare dataset")
                
            logging.info(f"Prepared {len(train_dataset)} training examples and {len(eval_dataset)} validation examples")
            
            # Create checkpoint directory if it doesn't exist
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                
            # Training loop
            best_eval_loss = float('inf')
            global_step = 0
            
            for epoch in range(num_epochs):
                self.model.model.train()
                total_loss = 0
                
                # Process data in batches
                for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                    batch = self.dataset_loader.get_batch(train_dataset, batch_size, i)
                    
                    # Forward pass
                    outputs = self.model.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    if (i + 1) % self.model.gradient_accumulation_steps == 0:
                        self.model.optimizer.step()
                        self.model.optimizer.zero_grad()
                        
                    total_loss += loss.item()
                    global_step += 1
                    
                    # Evaluation
                    if global_step % eval_steps == 0:
                        eval_loss, rouge_scores = self._evaluate(eval_dataset, batch_size)
                        
                        # Log evaluation metrics
                        log_message = f"Step {global_step} - Eval loss: {eval_loss:.4f}"
                        if rouge_scores:
                            log_message += f" - ROUGE-1: {rouge_scores.get('rouge1', 0):.4f} - ROUGE-2: {rouge_scores.get('rouge2', 0):.4f} - ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}"
                        logging.info(log_message)
                        
                        # Save checkpoint if best so far
                        if eval_loss < best_eval_loss and checkpoint_dir:
                            best_eval_loss = eval_loss
                            self.model.model.save_pretrained(f"{checkpoint_dir}/checkpoint-{global_step}")
                            self.model.tokenizer.save_pretrained(f"{checkpoint_dir}/checkpoint-{global_step}")
                            
                            # Save evaluation metrics
                            if rouge_scores:
                                import json
                                metrics_file = f"{checkpoint_dir}/checkpoint-{global_step}/metrics.json"
                                with open(metrics_file, "w") as f:
                                    json.dump({"eval_loss": eval_loss, **rouge_scores}, f, indent=2)
                            
                    # Regular checkpoint saving
                    if checkpoint_dir and global_step % save_steps == 0:
                        self.model.model.save_pretrained(f"{checkpoint_dir}/checkpoint-{global_step}")
                        self.model.tokenizer.save_pretrained(f"{checkpoint_dir}/checkpoint-{global_step}")
                
                avg_loss = total_loss / (len(train_dataset) / batch_size)
                logging.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save final model
            if checkpoint_dir:
                final_model_path = f"{checkpoint_dir}/final-model"
                self.model.model.save_pretrained(final_model_path)
                self.model.tokenizer.save_pretrained(final_model_path)
                logging.info(f"Final model saved to {final_model_path}")
            
            return {
                "final_loss": avg_loss, 
                "best_eval_loss": best_eval_loss,
                "final_model_path": final_model_path if checkpoint_dir else None
            }
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise

    def _evaluate(self, eval_dataset: Dataset, batch_size: int) -> tuple[float, dict[str, float]]:
        """Evaluate the model on validation data with ROUGE metrics.
        
        Args:
            eval_dataset: Dataset for evaluation
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (average evaluation loss, ROUGE scores dictionary)
        """
        self.model.model.eval()
        total_eval_loss = 0
        all_predictions = []
        all_references = []
        
        # Evaluate loss and generate predictions for ROUGE calculation
        with torch.no_grad():
            for i in range(0, len(eval_dataset), batch_size):
                batch = self.dataset_loader.get_batch(eval_dataset, batch_size, i)
                
                # Calculate loss
                outputs = self.model.model(**batch)
                total_eval_loss += outputs.loss.item()
                
                # Generate predictions for a subset of examples (to save time)
                if i % (5 * batch_size) == 0 and self.rouge is not None:
                    # Get input_ids and labels
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    
                    # Generate summaries
                    generated_ids = self.model.model.generate(
                        input_ids=input_ids,
                        max_length=128,
                        min_length=30,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                    
                    # Decode generated summaries and reference summaries
                    decoded_preds = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    decoded_labels = self.model.tokenizer.batch_decode(
                        [[l for l in label if l != -100] for label in labels], 
                        skip_special_tokens=True
                    )
                    
                    # Add to collections for ROUGE calculation
                    all_predictions.extend(decoded_preds)
                    all_references.extend(decoded_labels)
        
        # Calculate average loss
        avg_eval_loss = total_eval_loss / (len(eval_dataset) / batch_size)
        
        # Calculate ROUGE scores if we have predictions and the metric is available
        rouge_scores = {}
        if all_predictions and all_references and self.rouge is not None:
            try:
                # Compute ROUGE scores
                rouge_output = self.rouge.compute(
                    predictions=all_predictions,
                    references=all_references,
                    use_stemmer=True
                )
                
                # Extract scores
                rouge_scores = {
                    "rouge1": rouge_output["rouge1"].mid.fmeasure,
                    "rouge2": rouge_output["rouge2"].mid.fmeasure,
                    "rougeL": rouge_output["rougeL"].mid.fmeasure
                }
            except Exception as e:
                logging.error(f"Error computing ROUGE scores: {str(e)}")
        
        # Switch back to training mode
        self.model.model.train()
        
        return avg_eval_loss, rouge_scores