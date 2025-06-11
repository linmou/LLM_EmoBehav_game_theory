"""
This implementation may have bugs. 
Tensor parallel results are not consistent with single GPU results.
More details:
python -m unittest neuro_manipulation/repe/tests/test_sequence_prob_tp_consistency.py
"""

import torch
import numpy as np
import logging
import sys
from vllm import LLM, SamplingParams
from neuro_manipulation.model_layer_detector import ModelLayerDetector
import traceback
import torch.distributed as dist
from typing import List, Dict, Union, Optional
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Hook Function ---
def hook_fn_sequence_prob(module, args, output):
    """
    Forward hook function that captures logits for sequence probability calculation.
    It checks for state attached directly to the module instance.
    """
    print("!!! HOOK DEBUG: hook_fn_sequence_prob called!")
    print(f"!!! HOOK DEBUG: module is {module.__class__.__name__}")
    print(f"!!! HOOK DEBUG: output type is {type(output)}")

    if isinstance(output, tuple):
        print(f"!!! HOOK DEBUG: output is tuple with len {len(output)}")
        if len(output) > 0:
            print(f"!!! HOOK DEBUG: output[0] type is {type(output[0])}")
            if isinstance(output[0], torch.Tensor):
                print(f"!!! HOOK DEBUG: output[0] shape is {output[0].shape}")
    elif isinstance(output, torch.Tensor):
        print(f"!!! HOOK DEBUG: output shape is {output.shape}")
        
    # Check if state is set
    if not hasattr(module, '_sequence_prob_state'):
        print("!!! HOOK DEBUG: no _sequence_prob_state attribute found, returning.")
        return output
    
    state = module._sequence_prob_state
    target_sequences = state.get('target_sequences')
    tokenizer = state.get('tokenizer')
    tp_size = state.get('tp_size', 1)
    
    # Get rank from distributed context if available, otherwise use 0
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    
    if target_sequences is None or tokenizer is None:
        logger.warning(f"Rank {rank} - Module {module.__class__.__name__} - SequenceProb state incomplete, skipping.")
        return output

    try:
        # Identify the target tensor (logits from language model head)
        if isinstance(output, torch.Tensor):
            logits = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            logits = output[0]
        else:
            logger.error(f"Rank {rank} - Module {module.__class__.__name__} output type is not a Tensor or Tuple[Tensor, ...]: {type(output)}. Cannot capture logits.")
            return output

        logger.debug(f"Rank {rank} - Capturing logits from {module.__class__.__name__}. Logits shape: {logits.shape}, TP size: {tp_size}")

        # Store logits in state for later aggregation
        if 'captured_logits' not in state:
            state['captured_logits'] = []
        
        # Store logits with rank information for tensor parallel aggregation
        state['captured_logits'].append({
            'logits': logits.detach().clone(),
            'rank': rank,
            'shape': logits.shape
        })

        logger.debug(f"Rank {rank} - Logits captured and stored. Total captures: {len(state['captured_logits'])}")
        print(f"!!! HOOK DEBUG: Logits captured! Shape: {logits.shape}, Total captures: {len(state['captured_logits'])}")
        
        return output

    except Exception as e:
        logger.error(f"Rank {rank} - Error in SequenceProb hook for {module.__class__.__name__}: {e}", exc_info=True)
        print(f"!!! HOOK DEBUG: Exception in hook: {e}")
        return output

# --- RPC Functions ---
def _get_nested_module(model, module_path):
    """Helper to get a nested module by path."""
    modules = module_path.split('.')
    current_module = model
    for mod_name in modules:
        if hasattr(current_module, mod_name):
            current_module = getattr(current_module, mod_name)
        else:
             # Handle list-like access (e.g., model.layers[0])
             try:
                 idx = int(mod_name)
                 current_module = current_module[idx]
             except (ValueError, IndexError, TypeError):
                  logger.error(f"Could not find module part: {mod_name} in path {module_path}")
                  return None
    return current_module

def _find_lm_head_module(worker_self):
    """Finds the language model head module on the worker."""
    rank = worker_self.rank
    print(f"!!! RPC DEBUG: Worker Rank {rank} _find_lm_head_module called")
    
    if not hasattr(worker_self, 'model_runner') or not hasattr(worker_self.model_runner, 'model'):
        logger.error(f"RPC: Worker Rank {rank} could not find model_runner.model")
        print(f"!!! RPC DEBUG: Worker Rank {rank} could not find model_runner.model")
        return None
    model = worker_self.model_runner.model
    print(f"!!! RPC DEBUG: Worker Rank {rank} model type: {type(model)}")

    # Common language model head names
    lm_head_names = ['lm_head', 'language_model_head', 'head', 'output_projection', 'embed_out']
    print(f"!!! RPC DEBUG: Worker Rank {rank} available model attributes: {list(model.__dict__.keys())}")
    
    for head_name in lm_head_names:
        if hasattr(model, head_name):
            lm_head = getattr(model, head_name)
            logger.debug(f"RPC: Worker Rank {rank} found LM head: {head_name} - {lm_head.__class__.__name__}")
            print(f"!!! RPC DEBUG: Worker Rank {rank} found LM head: {head_name} - {lm_head.__class__.__name__}")
            return lm_head
    
    # If not found directly, try to find it in model structure
    # For some models, it might be nested deeper
    if hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        print(f"!!! RPC DEBUG: Worker Rank {rank} found LM head in model.model.lm_head")
        return model.model.lm_head
    
    logger.warning(f"RPC: Worker Rank {rank} - Could not find language model head. Available attributes: {list(model.__dict__.keys())}")
    print(f"!!! RPC DEBUG: Worker Rank {rank} - Could not find language model head. Available attributes: {list(model.__dict__.keys())}")
    return None

def _register_lm_head_hook_rpc(worker_self, hook_func):
    """RPC function to register a forward hook on the language model head."""
    rank = worker_self.rank
    try:
        logger.debug(f"Worker Rank {rank} attempting to find LM head module...")
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None:
            logger.error(f"RPC: Worker Rank {rank} failed to find LM head module.")
            return False

        logger.debug(f"Worker Rank {rank} found LM head: {lm_head.__class__.__name__}")
        logger.info(f"RPC: Worker Rank {rank} registering hook {hook_func.__name__} to {lm_head.__class__.__name__}")
        handle = lm_head.register_forward_hook(hook_func)

        if handle:
            logger.info(f"RPC: Worker Rank {rank} LM head hook registered successfully.")
            # Store the handle for potential removal
            if not hasattr(worker_self, '_lm_head_hook_handle'):
                 worker_self._lm_head_hook_handle = handle
            return True
        else:
            logger.error(f"RPC: Worker Rank {rank} LM head hook registration failed.")
            return False
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error during LM head hook registration: {e}", exc_info=True)
        return False

def _set_sequence_prob_state_rpc(worker_self, state):
    """RPC function to set the sequence probability state on the LM head module."""
    rank = worker_self.rank
    try:
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None:
            logger.error(f"RPC: Worker Rank {rank} failed to find LM head module to set state.")
            return False

        logger.debug(f"RPC: Worker Rank {rank} setting SequenceProb state on {lm_head.__class__.__name__}")
        # Attach state directly to the module instance
        lm_head._sequence_prob_state = state
        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error setting sequence probability state: {e}", exc_info=True)
        return False

def _reset_sequence_prob_state_rpc(worker_self):
    """RPC function to remove the sequence probability state from the LM head module."""
    rank = worker_self.rank
    try:
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None:
            logger.warning(f"RPC: Worker Rank {rank} could not find LM head module to reset state. Skipping.")
            return True

        if hasattr(lm_head, '_sequence_prob_state'):
            logger.debug(f"RPC: Worker Rank {rank} resetting SequenceProb state on {lm_head.__class__.__name__}")
            delattr(lm_head, '_sequence_prob_state')
        else:
            logger.debug(f"RPC: Worker Rank {rank} - No SequenceProb state found on {lm_head.__class__.__name__} to reset.")

        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error resetting sequence probability state: {e}", exc_info=True)
        return False

def _get_captured_logits_rpc(worker_self):
    """RPC function to retrieve captured logits from the LM head module."""
    rank = worker_self.rank
    try:
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None or not hasattr(lm_head, '_sequence_prob_state'):
            logger.warning(f"RPC: Worker Rank {rank} - No state found for logits retrieval.")
            print(f"!!! RPC DEBUG: Worker Rank {rank} - No state found for logits retrieval.")
            return None

        state = lm_head._sequence_prob_state
        captured_logits = state.get('captured_logits', [])
        
        logger.debug(f"RPC: Worker Rank {rank} - Retrieved {len(captured_logits)} captured logits")
        print(f"!!! RPC DEBUG: Worker Rank {rank} - Retrieved {len(captured_logits)} captured logits")
        print(f"!!! RPC DEBUG: Worker Rank {rank} - State keys: {list(state.keys())}")
        print(f"!!! RPC DEBUG: Worker Rank {rank} - State has _sequence_prob_state: {hasattr(lm_head, '_sequence_prob_state')}")
        return captured_logits
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error retrieving captured logits: {e}", exc_info=True)
        print(f"!!! RPC DEBUG: Worker Rank {rank} error retrieving captured logits: {e}")
        return None

# --- Main Class ---
class SequenceProbVLLMHook:
    def __init__(self, model: LLM, tokenizer, tensor_parallel_size: int = 1):
        """
        Initializes SequenceProbVLLMHook for capturing sequence probabilities.

        Args:
            model: The vLLM LLM instance.
            tokenizer: The tokenizer.
            tensor_parallel_size: The tensor parallel size used by the vLLM model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hook_registered = False
        self.tp_size = tensor_parallel_size

        if not hasattr(self.model, 'llm_engine') or not hasattr(self.model.llm_engine, 'collective_rpc'):
             raise AttributeError("Provided model does not have 'llm_engine.collective_rpc'. Is it a valid vLLM LLM object?")

        # Get tensor parallel size from engine if available
        if hasattr(self.model.llm_engine, 'parallel_config') and hasattr(self.model.llm_engine.parallel_config, 'tensor_parallel_size'):
            self.tp_size = self.model.llm_engine.parallel_config.tensor_parallel_size
            logger.info(f"Detected Tensor Parallel Size: {self.tp_size}")
        else:
            logger.warning("Could not detect tensor_parallel_size from engine's parallel_config. Using provided value.")

        logger.info(f"Initializing SequenceProbVLLMHook with TP size: {self.tp_size}")

        # Register the hook on the language model head
        self._register_lm_head_hook()

    def _register_lm_head_hook(self):
        """Register hook on the language model head across all workers."""
        logger.info("Registering LM head hook...")
        rpc_results = self.model.llm_engine.collective_rpc(
            _register_lm_head_hook_rpc,
            args=(hook_fn_sequence_prob,)
        )
        logger.info(f"RPC hook registration results: {rpc_results}")
        
        if not any(rpc_results):
            logger.error("Failed to register hook on any worker. Sequence probability capture will not work.")
            self.hook_registered = False
        else:
            self.hook_registered = True
            logger.info("LM head hook registered successfully.")

    def get_log_prob(self, 
                     text_inputs: List[str], 
                     target_sequences: List[str],
                     **kwargs) -> List[Dict[str, float]]:
        """
        Calculates log probabilities for target sequences given input prompts.

        Args:
            text_inputs: List of input prompts.
            target_sequences: List of target sequences to calculate probabilities for.
            **kwargs: Additional arguments for vLLM SamplingParams.

        Returns:
            List of dictionaries containing log probabilities for each target sequence.
            Each dict has keys: 'sequence', 'log_prob', 'prob', 'perplexity'
        """
        if not self.hook_registered:
            raise RuntimeError("Hook not registered. Cannot capture sequence probabilities.")

        results = []
        
        try:
            # Tokenize target sequences
            tokenized_targets = []
            for seq in target_sequences:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                # Also try with a leading space (common in GPT tokenizers)
                tokens_with_space = self.tokenizer.encode(' ' + seq, add_special_tokens=False)
                
                tokenized_targets.append({
                    'sequence': seq,
                    'tokens': tokens,
                    'token_ids': torch.tensor(tokens),
                    'tokens_with_space': tokens_with_space,
                    'token_ids_with_space': torch.tensor(tokens_with_space)
                })

            # Set state for capturing logits
            state = {
                'target_sequences': tokenized_targets,
                'tokenizer': self.tokenizer,
                'tp_size': self.tp_size,
                'captured_logits': []
            }

            # Set state via RPC
            logger.info("Setting sequence probability state...")
            rpc_results = self.model.llm_engine.collective_rpc(
                _set_sequence_prob_state_rpc,
                args=(state,)
            )
            
            if not all(rpc_results):
                logger.warning("Failed to set state on some workers.")

            # Prepare sampling parameters for generation
            # We need to generate enough tokens to cover our target sequence
            max_target_tokens = max(len(target['tokens']) for target in tokenized_targets)
            sampling_params = SamplingParams(
                max_tokens=kwargs.get('max_tokens', max_target_tokens),  # Generate enough for target
                temperature=0.0,  # Deterministic for probability calculation
                top_p=1.0,
                logprobs=20  # Max allowed in vLLM v1 is 20
            )

            # Run generation to capture logits
            logger.info("Running generation to capture logits...")
            logger.debug(f"About to call model.generate with sampling_params: {sampling_params}")
            outputs = self.model.generate(text_inputs, sampling_params)
            logger.debug(f"Generation completed successfully")
            
            # NEW APPROACH: Extract probabilities directly from vLLM output logprobs
            results = self._calculate_sequence_probabilities_from_output(
                outputs, tokenized_targets, text_inputs
            )

        finally:
            # Reset state
            logger.info("Resetting sequence probability state...")
            reset_results = self.model.llm_engine.collective_rpc(
                _reset_sequence_prob_state_rpc,
                args=()
            )
            if not all(reset_results):
                logger.warning("Failed to reset state on some workers.")

        return results

    def _calculate_sequence_probabilities_from_output(self, 
                                                    outputs: List,
                                                    tokenized_targets: List[Dict],
                                                    text_inputs: List[str]) -> List[Dict[str, float]]:
        """
        Calculate sequence probabilities directly from vLLM generation output logprobs.
        
        Args:
            outputs: vLLM generation outputs containing logprobs
            tokenized_targets: Tokenized target sequences
            text_inputs: Original input texts
            
        Returns:
            List of probability results for each target sequence
        """
        results = []
        
        try:
            if not outputs or len(outputs) == 0:
                logger.error("No generation outputs provided.")
                return []
            
            # For each target sequence, calculate its probability from the logprobs
            for target_info in tokenized_targets:
                sequence = target_info['sequence']
                token_ids = target_info['token_ids']
                
                # Extract logprobs from the first output (assuming single prompt)
                output = outputs[0]
                if not hasattr(output, 'outputs') or len(output.outputs) == 0:
                    logger.error(f"No completion outputs found for sequence '{sequence}'.")
                    continue
                
                completion_output = output.outputs[0]
                if not hasattr(completion_output, 'logprobs') or not completion_output.logprobs:
                    logger.error(f"No logprobs found for sequence '{sequence}'.")
                    continue
                
                # Calculate probability for the target sequence
                # Try both versions - with and without space
                log_prob = self._calculate_target_sequence_prob_from_logprobs(
                    completion_output.logprobs, token_ids
                )
                
                # If not found, try with space version
                if log_prob is None or log_prob.item() == 0.0:
                    token_ids_with_space = target_info['token_ids_with_space']
                    log_prob = self._calculate_target_sequence_prob_from_logprobs(
                        completion_output.logprobs, token_ids_with_space
                    )
                
                if log_prob is not None:
                    prob = torch.exp(log_prob).item()
                    perplexity = torch.exp(-log_prob).item()
                    
                    results.append({
                        'sequence': sequence,
                        'log_prob': log_prob.item(),
                        'prob': prob,
                        'perplexity': perplexity,
                        'num_tokens': len(token_ids)
                    })
                    
                    logger.info(f"Sequence '{sequence}': log_prob={log_prob.item():.4f}, prob={prob:.6f}, perplexity={perplexity:.4f}")
                else:
                    logger.warning(f"Could not calculate probability for sequence '{sequence}'.")

        except Exception as e:
            logger.error(f"Error calculating sequence probabilities from output: {e}", exc_info=True)
            
        return results

    def _calculate_target_sequence_prob_from_logprobs(self, 
                                                    logprobs_list: List[Dict], 
                                                    target_token_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Calculate log probability for a target sequence from vLLM logprobs output.
        
        Args:
            logprobs_list: List of logprobs dictionaries from vLLM output
            target_token_ids: Target token IDs tensor
            
        Returns:
            Log probability of the target sequence or None if calculation fails
        """
        try:
            total_log_prob = 0.0
            tokens_found = 0
            
            logger.debug(f"Looking for target tokens: {target_token_ids.tolist()}")
            logger.debug(f"Available positions: {len(logprobs_list)}")
            
            # For each token in our target sequence
            for target_idx, target_token_id in enumerate(target_token_ids):
                token_id_int = target_token_id.item()
                found_at_position = False
                
                # Check each position in the generated sequence
                for pos, logprobs_dict in enumerate(logprobs_list):
                    if token_id_int in logprobs_dict:
                        # Found target token at this position
                        logprob_info = logprobs_dict[token_id_int]
                        token_log_prob = logprob_info.logprob
                        total_log_prob += token_log_prob
                        tokens_found += 1
                        found_at_position = True
                        logger.debug(f"Found target token {token_id_int} at position {pos} with logprob {token_log_prob}")
                        break
                        
                if not found_at_position:
                    logger.debug(f"Target token {token_id_int} not found in any position")
                    
            logger.debug(f"Total log prob for target sequence: {total_log_prob}, tokens found: {tokens_found}/{len(target_token_ids)}")
            
            if tokens_found == 0:
                return None
                
            return torch.tensor(total_log_prob)
            
        except Exception as e:
            logger.error(f"Error calculating target sequence probability from logprobs: {e}", exc_info=True)
            return None

    def _calculate_sequence_probabilities(self, 
                                        all_captured_logits: List,
                                        tokenized_targets: List[Dict],
                                        text_inputs: List[str]) -> List[Dict[str, float]]:
        """
        Calculate sequence probabilities from captured logits.
        
        Args:
            all_captured_logits: Logits captured from all workers
            tokenized_targets: Tokenized target sequences
            text_inputs: Original input texts
            
        Returns:
            List of probability results for each target sequence
        """
        results = []
        
        try:
            # Aggregate logits across tensor parallel ranks if needed
            aggregated_logits = self._aggregate_logits_across_ranks(all_captured_logits)
            
            if aggregated_logits is None or len(aggregated_logits) == 0:
                logger.error("No valid logits captured.")
                return []

            # Calculate probabilities for each target sequence
            for target_info in tokenized_targets:
                sequence = target_info['sequence']
                token_ids = target_info['token_ids']
                
                log_prob = self._calculate_single_sequence_prob(aggregated_logits, token_ids)
                prob = torch.exp(log_prob).item()
                perplexity = torch.exp(-log_prob).item()
                
                results.append({
                    'sequence': sequence,
                    'log_prob': log_prob.item(),
                    'prob': prob,
                    'perplexity': perplexity,
                    'num_tokens': len(token_ids)
                })
                
                logger.info(f"Sequence '{sequence}': log_prob={log_prob.item():.4f}, prob={prob:.6f}, perplexity={perplexity:.4f}")

        except Exception as e:
            logger.error(f"Error calculating sequence probabilities: {e}", exc_info=True)
            
        return results

    def _aggregate_logits_across_ranks(self, all_captured_logits: List) -> Optional[torch.Tensor]:
        """
        Aggregate logits across tensor parallel ranks.
        
        Args:
            all_captured_logits: List of captured logits from all workers
            
        Returns:
            Aggregated logits tensor or None if aggregation fails
        """
        try:
            valid_logits = [logits for logits in all_captured_logits if logits is not None and len(logits) > 0]
            
            if not valid_logits:
                logger.error("No valid logits found from any worker.")
                return None

            # For tensor parallel, we need to concatenate logits along the vocabulary dimension
            if self.tp_size > 1:
                # Sort by rank to ensure correct order
                rank_logits = {}
                for worker_logits in valid_logits:
                    for capture in worker_logits:
                        rank = capture['rank']
                        if rank not in rank_logits:
                            rank_logits[rank] = []
                        rank_logits[rank].append(capture['logits'])
                
                # Concatenate logits from different ranks
                sorted_ranks = sorted(rank_logits.keys())
                logits_to_concat = []
                
                for rank in sorted_ranks:
                    # Use the first (most recent) capture from each rank
                    if len(rank_logits[rank]) > 0:
                        logits_to_concat.append(rank_logits[rank][-1])
                
                if len(logits_to_concat) > 1:
                    # Concatenate along vocabulary dimension (last dimension)
                    aggregated = torch.cat(logits_to_concat, dim=-1)
                    logger.debug(f"Aggregated logits from {len(logits_to_concat)} ranks. Final shape: {aggregated.shape}")
                    return aggregated
                else:
                    return logits_to_concat[0] if logits_to_concat else None
            else:
                # Single rank case - just use the logits from rank 0
                first_worker_logits = valid_logits[0]
                if len(first_worker_logits) > 0:
                    return first_worker_logits[-1]['logits']  # Most recent capture
                
        except Exception as e:
            logger.error(f"Error aggregating logits: {e}", exc_info=True)
            
        return None

    def _calculate_single_sequence_prob(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate log probability for a single sequence.
        
        Args:
            logits: Aggregated logits tensor [batch_size, seq_len, vocab_size]
            token_ids: Target token IDs [num_tokens]
            
        Returns:
            Log probability of the sequence
        """
        # Convert logits to probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Sum log probabilities for the target sequence
        sequence_log_prob = 0.0
        
        # We need to match the sequence position by position
        # Assuming logits are for the continuation of the input
        for i, token_id in enumerate(token_ids):
            if i < log_probs.shape[1]:  # Ensure we don't go out of bounds
                # Use the last batch item (assuming single input)
                token_log_prob = log_probs[-1, i, token_id.item()]
                sequence_log_prob += token_log_prob
                logger.debug(f"Token {i} (ID: {token_id.item()}): log_prob = {token_log_prob.item():.4f}")
        
        return torch.tensor(sequence_log_prob)

    def remove_hooks(self):
        """Remove registered hooks (placeholder for future implementation)."""
        logger.warning("Hook removal not yet implemented for SequenceProbVLLMHook.")
        pass


# --- Example Usage ---
if __name__ == "__main__":
    import torch
    import gc
    from transformers import AutoTokenizer

    # Configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    num_gpus = 1
    
    if torch.cuda.is_available():
         detected_gpus = torch.cuda.device_count()
         logger.info(f"CUDA available. Found {detected_gpus} GPUs.")
         num_gpus = min(num_gpus, detected_gpus)
    else:
         logger.error("CUDA not available. This example requires GPU.")
         sys.exit(1)

    prompt = "The capital of France is"
    target_sequences = ["Paris", "London", "Berlin"]

    llm = None
    try:
        # Initialize Model and Tokenizer
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading LLM {model_name} with tensor_parallel_size={num_gpus}...")
        llm = LLM(model=model_name,
                  tokenizer=tokenizer.name_or_path,
                  enforce_eager=True,
                  trust_remote_code=True,
                  tensor_parallel_size=num_gpus,
                  gpu_memory_utilization=0.85,
                  max_num_seqs=16)

        # Initialize Sequence Probability Hook
        logger.info("Initializing SequenceProbVLLMHook...")
        seq_prob_hook = SequenceProbVLLMHook(llm, tokenizer)

        # Calculate sequence probabilities
        logger.info("--- Calculating Sequence Probabilities ---")
        results = seq_prob_hook.get_log_prob([prompt], target_sequences)
        
        print("\n=== SEQUENCE PROBABILITY RESULTS ===")
        for result in results:
            print(f"Sequence: '{result['sequence']}'")
            print(f"  Log Probability: {result['log_prob']:.4f}")
            print(f"  Probability: {result['prob']:.6f}")
            print(f"  Perplexity: {result['perplexity']:.4f}")
            print(f"  Tokens: {result['num_tokens']}")
            print()

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"Traceback: {traceback.format_exc()}")

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if llm is not None:
            del llm
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleanup completed.")
