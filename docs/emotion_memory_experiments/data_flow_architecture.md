# Emotion Memory Experiments: Data Flow Architecture

## Overview

This document describes the complete data flow architecture for emotion memory experiments, with special focus on the relationship between PromptFormat and PromptWrapper components.

## Core Architecture Components

### 1. Model Setup & PromptFormat Creation

```
setup_model_and_tokenizer(repe_config, from_vllm=False)
    ↓
Creates model-specific PromptFormat instance:
    - Qwen2.5 → Qwen2.5PromptFormat  
    - Llama → LlamaPromptFormat
    - GPT → OpenAIPromptFormat
    - Other → BasePromptFormat
    ↓
Returns: (model, tokenizer, prompt_format, processor)
```

**Key Point**: Each model gets its own PromptFormat class that knows how to apply that model's specific chat template and tokenization requirements.

### 2. PromptWrapper Creation & Integration

```
get_memory_prompt_wrapper(task_type, prompt_format)
    ↓
Factory creates task-specific wrapper:
    - "passkey" → PasskeyPromptWrapper(prompt_format)
    - "conversational" → ConversationalQAPromptWrapper(prompt_format)  
    - "longbook" → LongContextQAPromptWrapper(prompt_format)
    - default → MemoryPromptWrapper(prompt_format)
    ↓
Each wrapper stores prompt_format for model-specific formatting
```

**Critical Relationship**: PromptWrapper takes PromptFormat in constructor and uses it to format prompts correctly for each model's tokenizer.

### 3. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMOTION MEMORY EXPERIMENT                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL SETUP                                   │
│                                                                             │
│  setup_model_and_tokenizer(repe_config)                                   │
│      ├── Load Model & Tokenizer                                           │
│      ├── Create PromptFormat (model-specific)                             │
│      │   ├── Qwen2.5PromptFormat                                          │
│      │   ├── LlamaPromptFormat                                            │
│      │   ├── OpenAIPromptFormat                                           │
│      │   └── BasePromptFormat                                             │
│      └── Load processor (for multimodal)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROMPT WRAPPER CREATION                          │
│                                                                             │
│  get_memory_prompt_wrapper(task_type, prompt_format)                      │
│      ├── Factory Pattern Selection:                                        │
│      │   ├── "passkey" → PasskeyPromptWrapper                             │
│      │   ├── "conversational" → ConversationalQAPromptWrapper             │
│      │   ├── "longbook" → LongContextQAPromptWrapper                      │
│      │   └── default → MemoryPromptWrapper                                │
│      │                                                                     │
│      └── Each wrapper.__init__(prompt_format)                             │
│          Stores model-specific formatting logic                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ADAPTER & DATASET                               │
│                                                                             │
│  get_adapter(config.benchmark)                                            │
│      ├── InfiniteBenchAdapter → InfiniteBenchDataset                      │
│      ├── LongBenchAdapter → LongBenchDataset                              │
│      └── LoCoMoAdapter → LoCoMoDataset                                     │
│                                                                             │
│  adapter.get_dataloader(prompt_wrapper=wrapper_partial)                   │
│      └── Dataset receives prompt_wrapper as parameter                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATASET.__GETITEM__                             │
│                                                                             │
│  For each item in dataset:                                                 │
│      ├── Load BenchmarkItem(id, input_text, context, ground_truth)        │
│      │                                                                     │
│      ├── Call prompt_wrapper(context=context, question=input_text)        │
│      │   │                                                                 │
│      │   ├── wrapper.system_prompt(context, question)                     │
│      │   │   └── Task-specific system message                             │
│      │   │                                                                 │
│      │   ├── wrapper.user_messages(user_messages)                         │
│      │   │   └── Process user instructions                                │
│      │   │                                                                 │
│      │   └── prompt_format.build(system_prompt, user_messages,            │
│      │                           enable_thinking=True/False)              │
│      │       └── 🔑 CRITICAL: Model-specific chat template applied        │
│      │                                                                     │
│      └── Return formatted dict:                                            │
│          {                                                                  │
│              "prompt": formatted_prompt,  # Model-ready text               │
│              "item": original_item,                                         │
│              "context": context,                                            │
│              "question": input_text,                                        │
│              "ground_truth": ground_truth,                                  │
│              "metadata": metadata                                           │
│          }                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATALOADER                                    │
│                                                                             │
│  DataLoader(dataset, batch_size=4, collate_fn=collate_memory_benchmarks)  │
│      └── Batches formatted items for efficient processing                  │
│                                                                             │
│  Batch Output:                                                              │
│  {                                                                          │
│      "prompt": [formatted_prompt_1, formatted_prompt_2, ...],              │
│      "items": [item_1, item_2, ...],                                       │
│      "contexts": [context_1, context_2, ...],                              │
│      "questions": [question_1, question_2, ...],                           │
│      "ground_truths": [gt_1, gt_2, ...],                                   │
│      "metadata": [meta_1, meta_2, ...]                                     │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EMOTION MANIPULATION                              │
│                                                                             │
│  rep_control_pipeline(                                                     │
│      batch["prompt"],  # ← Model-ready prompts with correct chat template │
│      activations=emotion_activations,                                      │
│      temperature=0.1,                                                      │
│      max_new_tokens=50                                                     │
│  )                                                                          │
│      └── Generates emotionally-influenced responses                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             EVALUATION                                     │
│                                                                             │
│  adapter.evaluate_response(response, ground_truth, task_type)              │
│      ├── InfiniteBench: Integer extraction, F1 score                      │
│      ├── LongBench: QA F1 score with normalization                        │
│      └── LoCoMo: F1 score with stemming                                    │
│                                                                             │
│  ResultRecord(                                                              │
│      emotion=emotion,                                                       │
│      intensity=intensity,                                                   │
│      prompt=formatted_prompt,                                               │
│      response=model_response,                                               │
│      score=evaluation_score                                                 │
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Critical PromptFormat ↔ PromptWrapper Relationship

### PromptFormat Responsibilities

```python
class PromptFormat(ABC):
    """Abstract base for model-specific prompt formatting"""
    
    @abstractmethod  
    def build(self, system_prompt: str, user_messages: List[str], 
              enable_thinking: bool = False) -> str:
        """Apply model's chat template and tokenization rules"""
        pass
```

**Examples:**
- **Qwen2.5**: `<|system|>\n{system}<|user|>\n{user}<|assistant|>\n`
- **Llama**: `<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{user} [/INST]`
- **OpenAI**: `[{"role": "system", "content": "{system}"}, {"role": "user", "content": "{user}"}]`

### PromptWrapper Responsibilities

```python
class MemoryPromptWrapper(PromptWrapper):
    """Task-specific prompt content creation"""
    
    def __init__(self, prompt_format: PromptFormat):
        self.prompt_format = prompt_format  # Store model formatter
    
    def system_prompt(self, context, question):
        """Create task-specific system message content"""
        return f"Answer this question based on context: {context}\nQuestion: {question}"
    
    def __call__(self, context, question, user_messages, enable_thinking=False):
        """Combine task content with model formatting"""
        return self.prompt_format.build(  # ← Use model's chat template
            self.system_prompt(context, question),
            self.user_messages(user_messages), 
            enable_thinking=enable_thinking
        )
```

## Example: Complete Flow for Single Item

### Input Data
```json
{
  "id": "passkey_1",
  "input": "What is the pass key?", 
  "context": "The pass key is 71432. Remember it...",
  "answer": "71432"
}
```

### Step 1: Model Setup
```python
# Creates Qwen2.5PromptFormat for this model
model, tokenizer, prompt_format, processor = setup_model_and_tokenizer(repe_config)
```

### Step 2: Wrapper Creation  
```python
# Creates PasskeyPromptWrapper with Qwen2.5PromptFormat
wrapper = get_memory_prompt_wrapper("passkey", prompt_format)
```

### Step 3: Dataset Processing
```python
# Dataset.__getitem__() calls:
prompt = wrapper(
    context="The pass key is 71432. Remember it...",
    question="What is the pass key?",
    user_messages="Please provide your answer.",
    enable_thinking=False
)
```

### Step 4: Wrapper Processing
```python
# PasskeyPromptWrapper.system_prompt():
system = "Find and return the passkey mentioned in the following text.\nText: The pass key is 71432..."

# PromptFormat.build() with Qwen2.5 template:
formatted_prompt = """<|system|>
Find and return the passkey mentioned in the following text.
Text: The pass key is 71432. Remember it...
Question: What is the pass key?

<|user|>
Please provide your answer.

<|assistant|>
"""
```

### Step 5: Model Generation
```python
# rep_control_pipeline receives model-ready prompt
response = rep_control_pipeline([formatted_prompt], activations=emotion_vectors)
# → "The pass key is 71432"
```

### Step 6: Evaluation
```python
# InfiniteBenchAdapter.evaluate_response()
score = adapter.evaluate_response("The pass key is 71432", "71432", "passkey")
# → 1.0 (correct integer extraction)
```

## Key Design Benefits

### 1. **Separation of Concerns**
- **PromptFormat**: Handles model-specific tokenization and chat templates
- **PromptWrapper**: Handles task-specific content and structure

### 2. **Model Portability**  
- Same MemoryPromptWrapper works with any model
- Just swap PromptFormat for different tokenizers
- No hardcoded chat templates in task logic

### 3. **Task Extensibility**
- Add new tasks by creating new PromptWrapper subclasses
- Reuse existing PromptFormat for model compatibility
- Consistent evaluation through adapter pattern

### 4. **Proper Integration**
- Follows emotion_game_experiment.py patterns exactly
- Seamless DataLoader integration
- Pipeline acceleration support

This architecture ensures that emotion memory experiments work correctly across different models while maintaining proper task-specific prompting and evaluation.