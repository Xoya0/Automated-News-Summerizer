import logging
import os
import argparse
from model import T5Summarizer
from train import ModelTrainer
from evaluation import ModelEvaluator
from gemini_integration import GeminiAIIntegration, HybridSummarizer

def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run T5 Summarizer pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'full', 'summarize'], default='full',
                        help='Pipeline mode: train, evaluate, full, or summarize (default: full)')
    parser.add_argument('--model_name', type=str, default='t5-small',
                        help='T5 model name (default: t5-small)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and evaluation (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training (default: 1e-4)')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Steps between evaluations during training (default: 100)')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Steps between model checkpoints (default: 500)')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directory to save model checkpoints and evaluation results (default: ../models)')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Number of samples to use for evaluation (default: 100)')
    parser.add_argument('--summarizer_mode', type=str, choices=['t5', 'gemini', 'hybrid'], default='hybrid',
                        help='Summarizer mode: t5, gemini, or hybrid (default: hybrid)')
    parser.add_argument('--input_text', type=str, default=None,
                        help='Text to summarize (for summarize mode)')
    parser.add_argument('--input_file', type=str, default=None,
                        help='File containing text to summarize (for summarize mode)')
    parser.add_argument('--summary_length', type=str, choices=['short', 'medium', 'long'], default='medium',
                        help='Summary length: short, medium, or long (default: medium)')
    parser.add_argument('--summary_tone', type=str, choices=['neutral', 'analytical', 'key_points'], default='neutral',
                        help='Summary tone: neutral, analytical, or key_points (default: neutral)')
    
    return parser.parse_args()

def run_training(args):
    """Run the training pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of training metrics
    """
    logging.info("Starting training pipeline")
    
    # Initialize model
    logging.info(f"Initializing T5 model: {args.model_name}")
    model = T5Summarizer(model_name=args.model_name, learning_rate=args.learning_rate, batch_size=args.batch_size)
    
    # Initialize trainer
    trainer = ModelTrainer(model)
    
    # Create output directory
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train model
    logging.info("Starting model training")
    train_metrics = trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        checkpoint_dir=checkpoint_dir,
        learning_rate=args.learning_rate
    )
    
    logging.info(f"Training complete. Metrics: {train_metrics}")
    return train_metrics

def run_evaluation(args, model=None):
    """Run the evaluation pipeline.
    
    Args:
        args: Command line arguments
        model: Optional pre-initialized model
        
    Returns:
        Dictionary of evaluation metrics
    """
    logging.info("Starting evaluation pipeline")
    
    # Initialize model if not provided
    if model is None:
        # Try to load the final model if it exists
        final_model_path = os.path.join(args.output_dir, 'checkpoints', 'final-model')
        if os.path.exists(final_model_path):
            logging.info(f"Loading trained model from {final_model_path}")
            model = T5Summarizer(model_name=final_model_path)
        else:
            logging.info(f"Initializing T5 model: {args.model_name}")
            model = T5Summarizer(model_name=args.model_name)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model)
    
    # Create evaluation output directory
    eval_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluate model
    logging.info(f"Evaluating model on {args.eval_samples} samples")
    eval_metrics = evaluator.evaluate_model(
        split='test',
        num_samples=args.eval_samples,
        output_dir=eval_dir
    )
    
    logging.info(f"Evaluation complete. Metrics: {eval_metrics}")
    
    # Run qualitative evaluation on a small subset
    logging.info("Running qualitative evaluation")
    evaluator.dataset_loader.load_dataset('test')
    qualitative_samples = evaluator.dataset_loader.get_batch(evaluator.dataset_loader.dataset, 5, 0)
    evaluator.qualitative_evaluation(
        examples=qualitative_samples,
        output_file=os.path.join(eval_dir, 'qualitative_evaluation.txt')
    )
    
    return eval_metrics

def run_summarization(args):
    """Run the summarization pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of summarization results
    """
    logging.info("Starting summarization pipeline")
    
    # Get input text
    if args.input_text:
        text = args.input_text
    elif args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Error reading input file: {str(e)}")
            return {"error": f"Could not read input file: {str(e)}"}
    else:
        logging.error("No input text or file provided")
        return {"error": "No input text or file provided"}
    
    # Initialize T5 model
    logging.info(f"Initializing T5 model: {args.model_name}")
    t5_model = T5Summarizer(model_name=args.model_name)
    
    # Initialize Gemini integration
    logging.info("Initializing Gemini AI integration")
    gemini_integration = GeminiAIIntegration()
    
    # Initialize hybrid summarizer
    hybrid_summarizer = HybridSummarizer(t5_model, gemini_integration)
    
    # Generate summary
    logging.info(f"Generating summary using {args.summarizer_mode} mode")
    summary_results = hybrid_summarizer.generate_summary(
        text=text,
        mode=args.summarizer_mode,
        length=args.summary_length,
        tone=args.summary_tone,
        post_process=True
    )
    
    # Create output directory for summaries
    summary_dir = os.path.join(args.output_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save summary results
    import json
    import time
    timestamp = int(time.time())
    output_file = os.path.join(summary_dir, f"summary_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(summary_results.get("final_summary", "No summary generated"))
    print("=" * 80 + "\n")
    
    logging.info(f"Summary saved to {output_file}")
    return summary_results

def main():
    """Run the complete pipeline."""
    # Set up logging
    setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run pipeline based on mode
    if args.mode == 'summarize':
        summary_results = run_summarization(args)
    else:
        if args.mode in ['train', 'full']:
            train_metrics = run_training(args)
        
        if args.mode in ['evaluate', 'full']:
            eval_metrics = run_evaluation(args)
    
    logging.info("Pipeline execution complete")

if __name__ == "__main__":
    main()