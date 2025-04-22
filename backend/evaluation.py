import logging
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_metric
from model import T5Summarizer
from dataset_loader import CNNDailyMailDataLoader

class ModelEvaluator:
    def __init__(self, model: T5Summarizer):
        """Initialize the model evaluator.
        
        Args:
            model: The T5Summarizer model to evaluate
        """
        self.model = model
        self.dataset_loader = CNNDailyMailDataLoader()
        
        # Initialize metrics
        try:
            self.rouge = load_metric('rouge')
            self.bleu = load_metric('bleu')
            logging.info("Evaluation metrics loaded successfully")
        except Exception as e:
            logging.error(f"Error loading evaluation metrics: {str(e)}")
            raise
    
    def evaluate_model(self, split: str = 'test', num_samples: Optional[int] = None, 
                      output_dir: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the model on the CNN/DailyMail dataset.
        
        Args:
            split: Dataset split to evaluate on ('test' or 'validation')
            num_samples: Number of samples to evaluate (None for all)
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load test data
            logging.info(f"Loading {split} data for evaluation")
            test_data = self.dataset_loader.load_dataset(split)
            
            if not test_data:
                raise ValueError(f"Failed to load {split} dataset")
            
            # Limit samples if specified
            if num_samples and num_samples < len(test_data):
                import random
                random.seed(42)  # For reproducibility
                test_data = random.sample(test_data, num_samples)
                logging.info(f"Sampled {num_samples} examples for evaluation")
            
            # Generate summaries and compute metrics
            references = []
            predictions = []
            
            logging.info(f"Generating summaries for {len(test_data)} examples")
            for i, example in enumerate(test_data):
                if i % 10 == 0:
                    logging.info(f"Processing example {i+1}/{len(test_data)}")
                
                # Get reference summary
                reference = example['summary']
                references.append(reference)
                
                # Generate prediction
                prediction = self.model.generate_summary(example['text'])
                predictions.append(prediction)
                
                # Save examples for qualitative review
                if output_dir and i < 10:  # Save first 10 examples
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, f"example_{i+1}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"ARTICLE:\n{example['text']}\n\n")
                        f.write(f"REFERENCE SUMMARY:\n{reference}\n\n")
                        f.write(f"GENERATED SUMMARY:\n{prediction}\n")
            
            # Compute ROUGE scores
            rouge_output = self.rouge.compute(predictions=predictions, references=references, use_stemmer=True)
            
            # Compute BLEU score
            tokenized_predictions = [pred.split() for pred in predictions]
            tokenized_references = [[ref.split()] for ref in references]  # BLEU expects list of list of references
            bleu_output = self.bleu.compute(predictions=tokenized_predictions, references=tokenized_references)
            
            # Combine metrics
            metrics = {
                "rouge1": rouge_output["rouge1"].mid.fmeasure,
                "rouge2": rouge_output["rouge2"].mid.fmeasure,
                "rougeL": rouge_output["rougeL"].mid.fmeasure,
                "bleu": bleu_output["bleu"]
            }
            
            # Save metrics if output directory is provided
            if output_dir:
                import json
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            
            logging.info(f"Evaluation complete. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Evaluation error: {str(e)}")
            raise
    
    def qualitative_evaluation(self, examples: List[Dict[str, str]], output_file: str) -> None:
        """Perform qualitative evaluation on a set of examples.
        
        Args:
            examples: List of dictionaries with 'text' and 'summary' keys
            output_file: File to save the evaluation results
        """
        try:
            results = []
            
            for i, example in enumerate(examples):
                text = example['text']
                reference = example['summary']
                
                # Generate summary
                prediction = self.model.generate_summary(text)
                
                # Add to results
                results.append({
                    "example_id": i+1,
                    "article": text,
                    "reference_summary": reference,
                    "generated_summary": prediction
                })
            
            # Save results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(f"Example {result['example_id']}:\n")
                    f.write(f"ARTICLE:\n{result['article'][:500]}...\n\n")
                    f.write(f"REFERENCE SUMMARY:\n{result['reference_summary']}\n\n")
                    f.write(f"GENERATED SUMMARY:\n{result['generated_summary']}\n\n")
                    f.write("-" * 80 + "\n\n")
            
            logging.info(f"Qualitative evaluation saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Qualitative evaluation error: {str(e)}")
            raise