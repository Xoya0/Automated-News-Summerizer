import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import re
import os
import nltk
from typing import Optional, List, Dict, Any
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

try:
    from language_tool_python import LanguageTool
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    logging.warning("LanguageTool not available. Advanced post-processing will be limited.")

log_file = os.path.join(os.path.dirname(__file__), 'summarizer.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

required_nltk_data = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
for data in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'taggers/{data}' if 'tagger' in data else f'corpora/{data}')
    except LookupError:
        nltk.download(data)

class T5Summarizer:
    """Wrapper for T5 model to generate text summaries with customizable parameters."""
    
    def __init__(self, model_name: str = "t5-small", learning_rate: float = 1e-4, batch_size: int = 8,
                 gradient_accumulation_steps: int = 4):
        """Initialize the T5 model and tokenizer with training configuration.
        
        Args:
            model_name: The name of the T5 model to use (default: t5-base)
            learning_rate: Learning rate for optimization (default: 1e-4)
            batch_size: Batch size for training (default: 8)
            gradient_accumulation_steps: Number of steps to accumulate gradients (default: 4)
        """
        try:
            logging.info("Initializing T5 Summarizer")
            
            cuda_available = torch.cuda.is_available()
            if cuda_available:
     
                torch.cuda.empty_cache()
                logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logging.info(f"Initial CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            self.device = torch.device("cuda" if cuda_available else "cpu")
            logging.info(f"Using device: {self.device}")
            
        
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=None)
                logging.info("Tokenizer loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load tokenizer: {str(e)}")
                raise
            
        
            try:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    cache_dir=None,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if cuda_available else torch.float32
                )
                self.model.to(self.device)
                
       
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    eps=1e-8,
                    weight_decay=0.01
                )
                
                # Training configuration
                self.batch_size = batch_size
                self.gradient_accumulation_steps = gradient_accumulation_steps
                self.global_step = 0
                
                logging.info("Model loaded successfully")
                if cuda_available:
                    logging.info(f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            except Exception as e:
                logging.error(f"Failed to load model: {str(e)}")
                if cuda_available:
                    torch.cuda.empty_cache()
                raise
            
            logging.info("T5 Summarizer initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize T5 model: {str(e)}"
            logging.error(error_msg)
            if cuda_available:
                torch.cuda.empty_cache()
            raise RuntimeError(error_msg)
        
    def get_wordnet_pos(self, tag: str) -> Optional[str]:
        """Map POS tag to WordNet POS tag."""
        tag_dict = {
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
            'NNP': wordnet.NOUN,
            'NNPS': wordnet.NOUN,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB
        }
        return tag_dict.get(tag, None)

    def generate_variations(self, sentence: str) -> List[str]:
        """Generate variations of a sentence using synonyms."""
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        variations = [sentence]  # Include original sentence

        for word, tag in tagged:
            wn_pos = self.get_wordnet_pos(tag)
            if not wn_pos:
                continue

            # Get synonyms
            synsets = wordnet.synsets(word, pos=wn_pos)
            synonyms = set()
            for synset in synsets:
                for lemma in synset.lemmas():
                    if lemma.name() != word and '_' not in lemma.name():
                        synonyms.add(lemma.name())

            # Create variations with synonyms
            for synonym in list(synonyms)[:2]:  # Limit to 2 synonyms per word
                new_words = words.copy()
                new_words[words.index(word)] = synonym
                variation = ' '.join(new_words)
                variations.append(variation)
                if len(variations) >= 3:  # Limit total variations
                    break

        return variations[:3]  # Return at most 3 variations

    def preprocess_text(self, text: str) -> str:
        """Preprocess the input text for summarization with optimized memory usage.
        
        Args:
            text: The input text to preprocess
            
        Returns:
            Preprocessed text ready for the model
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            # Process text in chunks to optimize memory usage
            chunk_size = 10000  # Process 10K characters at a time
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            processed_chunks = []

            for chunk in chunks:
                # Basic cleaning
                chunk = re.sub(r'\s+', ' ', chunk)
                
                # Remove common artifacts using regex patterns
                patterns = [
                    r'\([^)]*\) - ',  # (Reuters) -, (AP) -
                    r'\[[^\]]*\]',     # [VIDEO], [AUDIO], etc.
                    r'Share (?:Save|this article|on)',
                    r'(?:Follow us|Subscribe) (?:on|to)',
                    r'Read (?:more|Full Article):?',
                    r'(?:Photo|Image|Credit|Source):',
                    r'Advertisement|Sponsored Content'
                ]
                for pattern in patterns:
                    chunk = re.sub(pattern, '', chunk)
                
                processed_chunks.append(chunk)

            # Join processed chunks
            text = ' '.join(processed_chunks)
            
            # Efficient sentence processing
            sentences = nltk.sent_tokenize(text)
            
            # Process sentences in batches
            batch_size = 100
            processed_sentences = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                # Clean and normalize sentences
                batch = [re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', s) for s in batch]
                batch = [s.strip().capitalize() for s in batch if len(s.split()) >= 3]
                processed_sentences.extend(batch)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_sentences = []
            for s in processed_sentences:
                if s not in seen:
                    seen.add(s)
                    unique_sentences.append(s)
            
            # Build final text within token limit
            max_length = 512
            final_text = []
            current_length = 0
            
            for sentence in unique_sentences:
                tokens = self.tokenizer.encode(sentence)
                if current_length + len(tokens) < max_length:
                    final_text.append(sentence)
                    current_length += len(tokens)
                else:
                    break
            
            return ' '.join(final_text).strip()
            
        except Exception as e:
            logging.error(f"Error in text preprocessing: {str(e)}")
            # Return cleaned input if processing fails
            return re.sub(r'\s+', ' ', text).strip()
    
    def generate_summary(self, text: str, length: str = "medium", tone: str = "neutral", use_augmentation: bool = True,
                       decoding_strategy: str = "beam_search", sampling_params: Dict = None, 
                       post_process: bool = True, post_process_level: str = "basic") -> str:
        """Generate a summary of the input text with customizable parameters.
        
        Args:
            text: The input text to summarize
            length: The desired summary length ('short', 'medium', 'long')
            tone: The desired summary tone ('neutral', 'analytical', 'key_points')
            use_augmentation: Whether to use data augmentation techniques (default: True)
            decoding_strategy: The decoding strategy to use ('beam_search', 'top_k', 'nucleus')
            sampling_params: Additional parameters for the chosen decoding strategy
            post_process: Whether to apply post-processing to the summary (default: True)
            post_process_level: The level of post-processing to apply ('basic', 'medium', 'advanced')
            
        Returns:
            Generated summary text
        """
        # Preprocess the input text
        preprocessed_text = self.preprocess_text(text)
        
        if use_augmentation:
            # Generate variations for key sentences
            sentences = nltk.sent_tokenize(preprocessed_text)
            augmented_sentences = []
            
            # Process first few sentences for variations
            for sentence in sentences[:2]:  # Limit to first 2 sentences
                variations = self.generate_variations(sentence)
                augmented_sentences.extend(variations)
            
            # Combine original and augmented text
            preprocessed_text = ' '.join(augmented_sentences + sentences[2:])
        
        # Enhanced prompt engineering with specific instructions and examples
        task_instructions = {
            "neutral": "Generate a clear, concise, and well-structured summary that captures the main points. Focus on key information and maintain a balanced perspective. Format: ",
            "analytical": "Analyze the content critically and provide a detailed summary that examines key arguments, evidence, and implications. Include supporting details where relevant. Format: ",
            "key_points": "Extract and present the most important points in a clear, bullet-point style summary. Prioritize key findings, statistics, and crucial information. Format: "
        }
        
        # Add example summaries for better context
        example_formats = {
            "neutral": "The article discusses [topic]. The main points include [key points]. In conclusion, [summary].",
            "analytical": "This analysis examines [topic]. The key arguments presented are [points]. The evidence suggests [findings]. Important implications include [implications].",
            "key_points": "Key Points:\n- Main finding: [point]\n- Supporting evidence: [evidence]\n- Conclusion: [conclusion]"
        }
        
        # Prepare the input with enhanced prompting
        input_text = f"{task_instructions.get(tone, task_instructions['neutral'])}{preprocessed_text}"
        
        # Tokenize the input
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        # Set generation parameters based on desired length
        length_params = {
            "short": {"min_length": 40, "max_length": 75},
            "medium": {"min_length": 75, "max_length": 150},
            "long": {"min_length": 150, "max_length": 250}
        }
        
        # Set base generation parameters based on desired tone
        tone_params = {
            "neutral": {"no_repeat_ngram_size": 3, "length_penalty": 1.0},
            "analytical": {"no_repeat_ngram_size": 3, "length_penalty": 1.2},
            "key_points": {"no_repeat_ngram_size": 3, "length_penalty": 1.5}
        }
        
        # Configure decoding strategy parameters
        decoding_params = {}
        if sampling_params is None:
            sampling_params = {}
            
        if decoding_strategy == "beam_search":
            decoding_params.update({
                "num_beams": sampling_params.get("num_beams", 5),
                "early_stopping": True
            })
        elif decoding_strategy == "top_k":
            decoding_params.update({
                "do_sample": True,
                "top_k": sampling_params.get("top_k", 50),
                "temperature": sampling_params.get("temperature", 0.7)
            })
        elif decoding_strategy == "nucleus":
            decoding_params.update({
                "do_sample": True,
                "top_p": sampling_params.get("top_p", 0.9),
                "temperature": sampling_params.get("temperature", 0.8)
            })
        
        # Combine parameters
        generation_params = {
            **length_params.get(length, length_params["medium"]),
            **tone_params.get(tone, tone_params["neutral"]),
            "early_stopping": True
        }
        
        # Generate the summary with timeout handling
        try:
            # Use more efficient generation parameters for faster inference
            if decoding_strategy == "beam_search" and generation_params.get("num_beams", 5) > 3:
                # Reduce beam size for faster generation
                generation_params["num_beams"] = 3
            
            # Set a timeout for generation to prevent hanging (Windows-compatible)
            import threading
            import concurrent.futures
            
            # Define a function to generate summary
            def generate_with_model():
                return self.model.generate(inputs, **generation_params)
            
            # Use ThreadPoolExecutor with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate_with_model)
                try:
                    # Wait for up to 45 seconds for the result
                    summary_ids = future.result(timeout=45)
                except concurrent.futures.TimeoutError:
                    # Cancel the future if it times out
                    future.cancel()
                    raise TimeoutError("Model generation timed out after 45 seconds")
            
            # Decode the summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
        except TimeoutError as e:
            logging.error(f"Model generation timeout: {str(e)}")
            # Provide a fallback summary when timeout occurs
            return "The model took too long to generate a summary. Please try with a shorter text or different parameters."
        except Exception as e:
            logging.error(f"Error during summary generation: {str(e)}")
            # Provide a fallback summary when an error occurs
            return "An error occurred during summary generation. Please try again with different parameters."
        
        # Apply post-processing if requested
        if post_process:
            summary = self.post_process_summary(summary, post_process_level)
        
        return summary
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics including ROUGE, BLEU, and BERTScore.
        
        Args:
            eval_pred: Tuple containing predictions and labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        from evaluate import load
        rouge = load('rouge')
        bleu = load('bleu')
        bertscore = load('bertscore')
        
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up decoded text for better evaluation
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Compute ROUGE scores
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute BLEU score
        tokenized_preds = [pred.split() for pred in decoded_preds]
        tokenized_labels = [[label.split()] for label in decoded_labels]  # BLEU expects list of list of references
        bleu_result = bleu.compute(predictions=tokenized_preds, references=tokenized_labels)
        
        # Compute BERTScore (semantic similarity)
        try:
            bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            # Extract average precision, recall, and F1 from BERTScore
            bertscore_precision = sum(bertscore_result['precision']) / len(bertscore_result['precision'])
            bertscore_recall = sum(bertscore_result['recall']) / len(bertscore_result['recall'])
            bertscore_f1 = sum(bertscore_result['f1']) / len(bertscore_result['f1'])
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {str(e)}")
            bertscore_precision = 0.0
            bertscore_recall = 0.0
            bertscore_f1 = 0.0
        
        # Combine all metrics
        result = {
            **{f"rouge_{k}": v for k, v in rouge_result.items()},
            "bleu": bleu_result["bleu"],
            "bertscore_precision": bertscore_precision,
            "bertscore_recall": bertscore_recall,
            "bertscore_f1": bertscore_f1
        }
        
        return result
    
    # Training methods removed as we're only using the model for inference

    
    def post_process_summary(self, summary: str, level: str = "basic") -> str:
        """Apply post-processing to refine the generated summary.
        
        Args:
            summary: The raw summary text to process
            level: The level of post-processing to apply ('basic', 'medium', 'advanced')
            
        Returns:
            Refined summary text
        """
        if not summary:
            return ""
            
        logging.info(f"Post-processing summary with level: {level}")
        
        try:
            # Basic post-processing (always applied)
            processed_summary = summary.strip()
            
            # Fix common issues
            processed_summary = re.sub(r'\s+', ' ', processed_summary)  # Remove extra whitespace
            processed_summary = re.sub(r'\s+([.,;:!?])', r'\1', processed_summary)  # Fix spacing before punctuation
            processed_summary = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', processed_summary)  # Add space after punctuation
            
            # Remove duplicate punctuation
            processed_summary = re.sub(r'([.,;:!?])[.,;:!?]+', r'\1', processed_summary)
            
            # Ensure proper capitalization of sentences
            try:
                sentences = nltk.tokenize.sent_tokenize(processed_summary)
                sentences = [s[0].upper() + s[1:] if s and len(s) > 0 else s for s in sentences]
                processed_summary = ' '.join(sentences)
            except Exception as e:
                logging.warning(f"Sentence tokenization failed: {str(e)}")
            
            # Medium level post-processing
            if level in ["medium", "advanced"]:
                try:
                    # Fix common redundancies
                    redundancies = [
                        (r'\b(return|revert) back\b', r'\1'),
                        (r'\b(join|connect) together\b', r'\1'),
                        (r'\b(end|final) result\b', 'result'),
                        (r'\bbasic (essentials|fundamentals)\b', r'\1'),
                        (r'\badvance (forward|ahead)\b', 'advance'),
                        (r'\bcurrently (ongoing|happening)\b', r'\1'),
                        (r'\bpast (history|experience)\b', r'\1'),
                        (r'\bfuture plans\b', 'plans'),
                        (r'\brepeat again\b', 'repeat'),
                        (r'\bnew innovation\b', 'innovation')
                    ]
                    
                    for pattern, replacement in redundancies:
                        processed_summary = re.sub(pattern, replacement, processed_summary, flags=re.IGNORECASE)
                    
                    # Improve readability if available
                    try:
                        processed_summary = self._improve_readability(processed_summary)
                    except Exception as e:
                        logging.warning(f"Readability improvement failed: {str(e)}")
                    
                    # Apply grammar correction if LanguageTool is available
                    if LANGUAGE_TOOL_AVAILABLE:
                        try:
                            language_tool = LanguageTool('en-US')
                            processed_summary = language_tool.correct(processed_summary)
                        except Exception as e:
                            logging.warning(f"Grammar correction failed: {str(e)}")
                    
                except Exception as e:
                    logging.warning(f"Medium-level post-processing failed: {str(e)}")
            
            # Advanced level post-processing
            if level == "advanced":
                try:
                    # Fix passive voice (when appropriate)
                    passive_patterns = [
                        (r'\bis being ([a-z]+ed)\b', r'is \1'),
                        (r'\bwas being ([a-z]+ed)\b', r'was \1'),
                        (r'\bare being ([a-z]+ed)\b', r'are \1'),
                        (r'\bwere being ([a-z]+ed)\b', r'were \1'),
                        (r'\bhas been ([a-z]+ed)\b', r'has \1'),
                        (r'\bhave been ([a-z]+ed)\b', r'have \1')
                    ]
                    
                    for pattern, replacement in passive_patterns:
                        processed_summary = re.sub(pattern, replacement, processed_summary, flags=re.IGNORECASE)
                    
                    # Try to restructure sentences if available
                    if hasattr(self, '_restructure_sentences'):
                        try:
                            processed_summary = self._restructure_sentences(processed_summary)
                        except Exception as e:
                            logging.warning(f"Sentence restructuring failed: {str(e)}")
                    
                    # Try to ensure consistency if available
                    if hasattr(self, '_ensure_consistency'):
                        try:
                            processed_summary = self._ensure_consistency(processed_summary)
                        except Exception as e:
                            logging.warning(f"Consistency enforcement failed: {str(e)}")
                    
                    # Ensure proper paragraph breaks for longer summaries
                    if len(processed_summary) > 300:
                        try:
                            sentences = sent_tokenize(processed_summary)
                            if len(sentences) >= 6:
                                # Group sentences into paragraphs (approximately 3-4 sentences per paragraph)
                                paragraphs = []
                                paragraph_size = max(3, len(sentences) // 3)  # Aim for 3 paragraphs
                                
                                for i in range(0, len(sentences), paragraph_size):
                                    paragraph = ' '.join(sentences[i:i+paragraph_size])
                                    paragraphs.append(paragraph)
                                
                                processed_summary = '\n\n'.join(paragraphs)
                        except Exception as e:
                            logging.warning(f"Paragraph restructuring failed: {str(e)}")
                    
                except Exception as e:
                    logging.warning(f"Advanced-level post-processing failed: {str(e)}")
            
            logging.info("Post-processing completed successfully")
            return processed_summary
            
        except Exception as e:
            logging.error(f"Post-processing failed completely: {str(e)}")
            # Return the original summary if post-processing fails
            return summary.strip()
    
    def _improve_readability(self, text: str) -> str:
        """Improve the readability of the text by simplifying vocabulary and sentence structure.
        
        Args:
            text: The text to improve
            
        Returns:
            Text with improved readability
        """
        # Replace complex words with simpler alternatives
        complex_word_replacements = {
            "utilize": "use",
            "implement": "use",
            "facilitate": "help",
            "endeavor": "try",
            "commence": "begin",
            "terminate": "end",
            "subsequently": "later",
            "additionally": "also",
            "numerous": "many",
            "obtain": "get",
            "demonstrate": "show",
            "sufficient": "enough",
            "approximately": "about",
            "initiate": "start",
            "finalize": "finish",
            "prioritize": "focus on",
            "necessitate": "need",
            "endeavour": "try",
            "ascertain": "find out",
            "conceptualize": "think of"
        }
        
        for complex_word, simple_word in complex_word_replacements.items():
            text = re.sub(r'\b' + complex_word + r'\b', simple_word, text, flags=re.IGNORECASE)
        
        # Break up very long sentences
        sentences = sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 25:  # If sentence is too long
                # Try to split at conjunctions or transition phrases
                split_patterns = [
                    r'(, and |, but |, however |, therefore |, moreover |, furthermore |, consequently |, nevertheless |, in addition |, as a result )',
                    r'(\. Additionally, |\. Furthermore, |\. Moreover, |\. However, |\. Therefore, )'
                ]
                
                for pattern in split_patterns:
                    parts = re.split(pattern, sentence)
                    if len(parts) > 1:
                        new_parts = []
                        for i in range(0, len(parts), 2):
                            if i+1 < len(parts):
                                new_parts.append(parts[i] + parts[i+1].strip())
                            else:
                                new_parts.append(parts[i])
                        processed_sentences.extend(new_parts)
                        break
                else:  # If no split was made with the patterns
                    # Try to split at a reasonable point (around the middle)
                    if len(words) > 30:
                        mid_point = len(words) // 2
                        # Find the nearest period, comma, or semicolon to the midpoint
                        for i in range(mid_point, mid_point - 10, -1):
                            if i > 0 and i < len(words) and any(words[i].endswith(p) for p in ['.', ',', ';']):
                                first_part = ' '.join(words[:i+1])
                                second_part = ' '.join(words[i+1:])
                                if not second_part.strip():
                                    processed_sentences.append(first_part)
                                else:
                                    processed_sentences.append(first_part)
                                    processed_sentences.append(second_part)
                                break
                        else:
                            processed_sentences.append(sentence)
                    else:
                        processed_sentences.append(sentence)
            else:
                processed_sentences.append(sentence)
        
        # Ensure proper capitalization after splitting
        for i in range(len(processed_sentences)):
            if processed_sentences[i] and not processed_sentences[i][0].isupper():
                processed_sentences[i] = processed_sentences[i][0].upper() + processed_sentences[i][1:]
        
        return ' '.join(processed_sentences)
    
    def _restructure_sentences(self, text: str) -> str:
        """Restructure sentences for better flow and coherence.
        
        Args:
            text: The text to restructure
            
        Returns:
            Text with improved sentence flow and coherence
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 1:
            return text
            
        # Define transition categories and their corresponding words
        transitions = {
            'addition': ["Additionally", "Furthermore", "Moreover", "In addition", "Also"],
            'contrast': ["However", "Nevertheless", "Conversely", "On the other hand", "Yet"],
            'cause_effect': ["Therefore", "Consequently", "As a result", "Thus", "Hence"],
            'emphasis': ["Indeed", "Notably", "Importantly", "Significantly"],
            'example': ["For example", "For instance", "Specifically", "To illustrate"],
            'conclusion': ["In conclusion", "To summarize", "Finally", "In summary", "Ultimately"]
        }
        
        # Analyze sentence relationships and add appropriate transitions
        restructured_sentences = [sentences[0]]  # Keep first sentence as is
        
        for i in range(1, len(sentences)):
            current = sentences[i]
            previous = sentences[i-1]
            
            # Skip if the sentence already starts with a transition word
            if any(current.startswith(word) for transition_type in transitions.values() for word in transition_type):
                restructured_sentences.append(current)
                continue
                
            # Determine relationship between sentences to choose appropriate transition
            if i == len(sentences) - 1:  # Last sentence
                transition_type = 'conclusion'
            elif any(word in current.lower() for word in ["but", "however", "although", "though", "despite", "yet"]):
                transition_type = 'contrast'
            elif any(word in current.lower() for word in ["therefore", "thus", "because", "since", "so"]):
                transition_type = 'cause_effect'
            elif any(word in current.lower() for word in ["example", "instance", "illustrate", "demonstrate"]):
                transition_type = 'example'
            elif any(word in current.lower() for word in ["important", "significant", "notably", "essential"]):
                transition_type = 'emphasis'
            else:
                transition_type = 'addition'
                
            # Only add transitions to about 30% of sentences to avoid overuse
            if i % 3 == 0 or i == len(sentences) - 1:  # Add to every third sentence and the last one
                import random
                transition = random.choice(transitions[transition_type])
                current = transition + ", " + current[0].lower() + current[1:]
            
            restructured_sentences.append(current)
        
        return ' '.join(restructured_sentences)
    
    def _ensure_consistency(self, text: str) -> str:
        """Ensure consistency in terminology throughout the text.
        
        Args:
            text: The text to process for terminology consistency
            
        Returns:
            Text with consistent terminology
        """
        # Find potential entity variations (e.g., "AI" and "artificial intelligence")
        words = word_tokenize(text.lower())
        word_freq = {}
        
        # Count word frequencies
        for word in words:
            if word.isalnum() and len(word) > 2:  # Skip short words and punctuation
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Identify common acronyms and their expanded forms
        acronyms = {}
        expanded_forms = {}
        
        # Pattern for potential acronyms (all caps, 2-6 letters)
        acronym_pattern = re.compile(r'\b[A-Z]{2,6}\b')
        potential_acronyms = acronym_pattern.findall(text)
        
        # Find expanded forms for acronyms
        for acronym in potential_acronyms:
            # Look for pattern like "Artificial Intelligence (AI)" or "AI (Artificial Intelligence)"
            expanded_pattern1 = re.compile(r'([A-Z][a-z]+\s)+\(' + acronym + r'\)', re.IGNORECASE)
            expanded_pattern2 = re.compile(acronym + r'\s+\(([A-Z][a-z]+\s)+\)', re.IGNORECASE)
            
            expanded_match1 = expanded_pattern1.search(text)
            expanded_match2 = expanded_pattern2.search(text)
            
            if expanded_match1:
                expanded_form = expanded_match1.group(0).split('(')[0].strip()
                acronyms[acronym.lower()] = expanded_form.lower()
                expanded_forms[expanded_form.lower()] = acronym.lower()
            elif expanded_match2:
                expanded_form = expanded_match2.group(0).split('(')[1].rstrip(')')
                acronyms[acronym.lower()] = expanded_form.lower()
                expanded_forms[expanded_form.lower()] = acronym.lower()
        
        # Standardize terminology based on frequency
        processed_text = text
        
        # Replace less frequent variations with more frequent ones
        for word in word_freq:
            if word_freq[word] >= 2:  # Only consider words that appear multiple times
                # Look for potential variations
                for other_word in word_freq:
                    # Skip comparing the same word or very different words
                    if other_word == word or abs(len(other_word) - len(word)) > 5:
                        continue
                        
                    # Check for similarity: starting with same letter or one is substring of other
                    if (other_word.startswith(word[0]) or word.startswith(other_word[0]) or 
                        word in other_word or other_word in word):
                        # If one is significantly more frequent, replace the less frequent one
                        if word_freq[word] > 2 * word_freq[other_word]:
                            # Use word boundary in regex to avoid partial word replacements
                            processed_text = re.sub(r'\b' + other_word + r'\b', word, processed_text, flags=re.IGNORECASE)
        
        # Standardize acronym usage - choose either all acronyms or all expanded forms based on frequency
        for acronym, expanded in acronyms.items():
            # Determine which form is more frequent
            acronym_count = word_freq.get(acronym, 0)
            # For expanded forms, count the sum of individual words
            expanded_words = expanded.split()
            expanded_count = sum(word_freq.get(w, 0) for w in expanded_words) / len(expanded_words)
            
            # If acronym is more frequent, replace expanded forms with acronym
            if acronym_count > expanded_count:
                # Replace expanded form with acronym (first occurrence keeps both)
                first_occurrence = True
                for match in re.finditer(r'\b' + re.escape(expanded) + r'\b', processed_text, re.IGNORECASE):
                    if first_occurrence:
                        # Keep the first occurrence as is (with both forms)
                        first_occurrence = False
                        continue
                    # Replace subsequent occurrences with acronym
                    start, end = match.span()
                    processed_text = processed_text[:start] + acronym.upper() + processed_text[end:]
            # If expanded form is more frequent, replace acronyms with expanded form
            elif expanded_count > acronym_count:
                # Replace acronym with expanded form (first occurrence keeps both)
                first_occurrence = True
                for match in re.finditer(r'\b' + acronym + r'\b', processed_text, re.IGNORECASE):
                    if first_occurrence:
                        # Keep the first occurrence as is (with both forms)
                        first_occurrence = False
                        continue
                    # Replace subsequent occurrences with expanded form
                    start, end = match.span()
                    processed_text = processed_text[:start] + expanded.capitalize() + processed_text[end:]
        
        return processed_text
    
    
    
    def _clean_training_data(self, data: List[Dict[str, str]], noise_threshold: float) -> List[Dict[str, str]]:
        """Clean training data by removing noisy examples."""
        cleaned_data = []
        for example in data:
            # Basic validation
            if not example.get('text') or not example.get('summary'):
                continue
                
            # Remove examples with very short or very long text/summaries
            if len(example['text'].split()) < 10 or len(example['summary'].split()) < 3:
                continue
            if len(example['text'].split()) > 1000 or len(example['summary'].split()) > 150:
                continue
                
            # Remove examples with high similarity between text and summary
            text_tokens = set(example['text'].lower().split())
            summary_tokens = set(example['summary'].lower().split())
            overlap = len(text_tokens.intersection(summary_tokens)) / len(summary_tokens)
            if overlap > noise_threshold:
                continue
                
            cleaned_data.append(example)
        return cleaned_data
        
    def _prepare_dataset(self, data: List[Dict[str, str]]) -> Any:
        """Prepare dataset for training or evaluation.
        
        Args:
            data: List of dictionaries containing text and summary pairs
            
        Returns:
            Dataset object ready for training/evaluation
        """
        from datasets import Dataset
        import numpy as np
        
        try:
            # Convert list of dicts to Dataset object
            dataset = Dataset.from_dict({
                'text': [item['text'] for item in data],
                'summary': [item['summary'] for item in data]
            })
            
            # Tokenize function for batch processing
            def tokenize_function(examples):
                # Tokenize inputs
                model_inputs = self.tokenizer(
                    examples['text'],
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Tokenize targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        examples['summary'],
                        max_length=150,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                
                model_inputs['labels'] = labels['input_ids']
                return model_inputs
            
            # Apply tokenization to the dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            return tokenized_dataset
            
        except Exception as e:
            logging.error(f"Error preparing dataset: {str(e)}")
            raise



def fine_tune(self, train_data: List[Dict[str, str]],
              eval_data: Optional[List[Dict[str, str]]] = None,
              output_dir: str = "./fine_tuned_model",
              num_train_epochs: int = 3,
              clean_data: bool = True,
              noise_threshold: float = 0.8,
              learning_rate: float = 1e-4,
              warmup_steps: int = 500) -> Dict[str, Any]:
    """Fine-tune the T5 model on custom data with optional data cleaning.

    Args:
        train_data: List of dictionaries with 'text' and 'summary' keys
        eval_data: Optional evaluation data in the same format as train_data
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        clean_data: Whether to apply data cleaning to remove noisy examples
        noise_threshold: Threshold for identifying noisy examples (higher = more strict)
        learning_rate: Learning rate for optimization (default: 1e-4)
        warmup_steps: Number of warmup steps for learning rate scheduler

    Returns:
        Dictionary containing training metrics and model information
    """
    try:
        if not train_data or not isinstance(train_data, list):
            raise ValueError("Training data must be a non-empty list of examples")

        for example in train_data[:5]:
            if not isinstance(example, dict) or 'text' not in example or 'summary' not in example:
                raise ValueError("Each training example must be a dictionary with 'text' and 'summary' keys")

        logging.info(f"Starting fine-tuning process with {len(train_data)} training examples")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Clean the training data if requested
        if clean_data:
            logging.info(f"Cleaning training data with noise threshold: {noise_threshold}")
            train_data = self._clean_training_data(train_data, noise_threshold)
            logging.info(f"After cleaning: {len(train_data)} training examples remain")
            if not train_data:
                raise ValueError("No training examples remain after cleaning. Consider lowering the noise threshold.")

        # Prepare the datasets
        logging.info("Preparing datasets for training")
        train_dataset = self._prepare_dataset(train_data)
        if not train_dataset or len(train_dataset) == 0:
            raise ValueError("Failed to prepare training dataset")

        eval_dataset = None
        if eval_data:
            logging.info(f"Preparing evaluation dataset with {len(eval_data)} examples")
            eval_dataset = self._prepare_dataset(eval_data)
            if not eval_dataset or len(eval_dataset) == 0:
                logging.warning("Failed to prepare evaluation dataset, proceeding without evaluation")
                eval_dataset = None

        # Define training arguments with enhanced configuration
        logging.info("Configuring training arguments")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            save_total_limit=2,  # Only keep the 2 best checkpoints
            eval_steps=500 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="rouge1" if eval_dataset else None,
            greater_is_better=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            warmup_steps=warmup_steps,
            report_to="none",  # Disable wandb/tensorboard reporting
            dataloader_num_workers=4 if torch.cuda.is_available() else 0  # Parallel data loading
        )

        # Initialize the Trainer
        logging.info("Initializing trainer")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            tokenizer=self.tokenizer  # Provide tokenizer for better logging
        )

        # Start training with progress tracking
        logging.info("Starting training...")
        train_result = trainer.train()
        train_metrics = train_result.metrics

        # Log training metrics
        logging.info(f"Training metrics: {train_metrics}")

        # Run final evaluation
        final_metrics = {}
        if eval_dataset:
            logging.info("Running final evaluation")
            eval_metrics = trainer.evaluate()
            final_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
            logging.info(f"Evaluation metrics: {final_metrics}")

        # Save the fine-tuned model
        logging.info(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training arguments for future reference
        with open(f"{output_dir}/training_args.json", "w") as f:
            import json
            json.dump(training_args.to_dict(), f, indent=2)

        logging.info("Fine-tuning completed successfully")

        # Return metrics and model information
        return {
            "training_metrics": train_metrics,
            "evaluation_metrics": final_metrics,
            "model_path": output_dir,
            "num_examples": len(train_dataset),
            "num_epochs": num_train_epochs,
            "training_time": train_result.metrics.get("train_runtime", 0)
        }
    except Exception as e:
        error_msg = f"Fine-tuning failed: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    
    def _calculate_noise_score(self, text: str, summary: str) -> float:
        """Calculate a noise score for a text-summary pair.
        
        Args:
            text: The input text
            summary: The summary text
            
        Returns:
            Noise score (0-1, lower is better)
        """
        # Initialize noise score
        noise_score = 0.0
        
        # Check if summary is too long relative to text
        if len(summary) > len(text) * 0.8:
            noise_score += 0.3
        
        # Check if summary is too short
        if len(summary) < 10:
            noise_score += 0.3
        
        # Check for high symbol-to-text ratio (potential noise)
        symbol_ratio = sum(1 for c in summary if c in string.punctuation) / max(1, len(summary))
        if symbol_ratio > 0.3:
            noise_score += 0.2
            
        # Check for HTML or code fragments
        if re.search(r'<[^>]+>|\{\{|\}\}|\[\[|\]\]', summary):
            noise_score += 0.4
            
        # Check for URLs or file paths
        if re.search(r'https?://|www\.|/\w+/\w+|\w+\.\w+\.\w+', summary):
            noise_score += 0.3
        
        # Check for repetitive patterns
        words = summary.lower().split()
        
        # Check for non-English content
        try:
            # Use NLTK to detect if the text contains mostly English words
            english_words = set(word.lower() for word in nltk.corpus.words.words())
            non_english_ratio = sum(1 for word in words if word.isalpha() and word not in english_words) / max(1, len(words))
            if non_english_ratio > 0.5:  # If more than 50% words are not in English dictionary
                noise_score += 0.3
        except Exception:
            # If there's an error in language detection, don't penalize
            pass
        unique_words = set(words)
        if len(words) > 0 and len(unique_words) / len(words) < 0.5:
            noise_score += 0.2
        
        # Check if summary contains common spam phrases
        spam_phrases = ["click here", "buy now", "free offer", "limited time", "subscribe"]
        if any(phrase in summary.lower() for phrase in spam_phrases):
            noise_score += 0.3
        
        return min(noise_score, 1.0)  # Cap at 1.0
    
    def _prepare_dataset(self, data: List[Dict[str, str]], is_cnn_dailymail: bool = False) -> Dataset:
        """Prepare a dataset for training or evaluation with enhanced preprocessing.
        
        Args:
            data: List of dictionaries with 'text' and 'summary' keys
            is_cnn_dailymail: Flag indicating if the data is from CNN/DailyMail dataset
            
        Returns:
            HuggingFace Dataset ready for training or evaluation
        """
        if not data:
            logging.warning("Empty dataset provided for preparation")
            return None
            
        # Convert to format expected by T5
        processed_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        logging.info(f"Preparing dataset with {len(data)} examples")
        
        for i, example in enumerate(data):
            text = example.get('text', '')
            summary = example.get('summary', '')
            
            # Skip invalid examples
            if not text or not summary:
                continue
                
            # Apply additional preprocessing for better training
            # 1. Ensure text is properly formatted
            text = text.strip()
            
            # 2. Ensure summary is properly formatted
            summary = summary.strip()
            
            # 3. Add task-specific prefix for T5
            prefix = "summarize news: " if is_cnn_dailymail else "summarize: "
            input_text = f"{prefix}{text}"
            
            # 4. Tokenize with appropriate padding and truncation
            try:
                input_encodings = self.tokenizer(input_text, max_length=512, padding="max_length", truncation=True)
                target_encodings = self.tokenizer(summary, max_length=128, padding="max_length", truncation=True)
                
                # 5. Add to processed data
                processed_data["input_ids"].append(input_encodings["input_ids"])
                processed_data["attention_mask"].append(input_encodings["attention_mask"])
                processed_data["labels"].append(target_encodings["input_ids"])
                
                # Log progress for large datasets
                if (i+1) % 1000 == 0:
                    logging.info(f"Processed {i+1}/{len(data)} examples for dataset preparation")
                    
            except Exception as e:
                logging.warning(f"Error processing example {i}: {str(e)}")
                continue
        
        # Verify we have data to process
        if not processed_data["input_ids"]:
            logging.error("No valid examples found after preprocessing")
            return None
            
        # Convert to Dataset
        dataset = Dataset.from_dict(processed_data)
        logging.info(f"Dataset preparation complete: {len(dataset)} examples ready for training")
        return dataset
    
    def batch_summarize(self, texts: List[str], length: str = "medium", tone: str = "neutral", 
                       post_process: bool = True, post_process_level: str = "basic") -> List[str]:
        """Generate summaries for a batch of texts with optional post-processing.
        
        Args:
            texts: List of input texts to summarize
            length: The desired summary length ('short', 'medium', 'long')
            tone: The desired summary tone ('neutral', 'analytical', 'key_points')
            post_process: Whether to apply post-processing to the summaries
            post_process_level: The level of post-processing to apply ('basic', 'medium', 'advanced')
            
        Returns:
            List of generated summaries
        """
        self.model.eval()
        summaries = [self.generate_summary(text, length, tone) for text in texts]
        
        if post_process:
            summaries = [self.post_process_summary(summary, post_process_level) for summary in summaries]
            
        return summaries