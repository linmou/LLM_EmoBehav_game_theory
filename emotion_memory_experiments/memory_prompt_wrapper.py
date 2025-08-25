"""
Memory benchmark prompt wrapper following GameScenarioDataset pattern.
Integrates with neuro_manipulation.prompt_wrapper.PromptWrapper for proper model-specific formatting.
"""

from typing import Any, Dict, Optional, Union

from neuro_manipulation.prompt_formats import PromptFormat
from neuro_manipulation.prompt_wrapper import PromptWrapper


class MemoryPromptWrapper(PromptWrapper):
    """
    Prompt wrapper for memory benchmark tasks.
    Follows the pattern from GameScenarioDataset but adapted for memory benchmarks.
    """

    system_prompt_format = "You are a helpful AI assistant. Please answer the following question based on the given context."

    def __init__(self, prompt_format: PromptFormat):
        super().__init__(prompt_format)

    def system_prompt(self, context, question):
        """Create system prompt for memory tasks"""
        if context:
            return f"{self.system_prompt_format}\n\nContext: {context}\n\nQuestion: {question}"
        else:
            return f"{self.system_prompt_format}\n\nQuestion: {question}"

    def user_messages(self, user_messages):
        """Process user messages (inherited from parent)"""
        if type(user_messages) == str:
            return [user_messages]
        return user_messages

    def augment_context(
        self,
        context: Optional[str],
        augmentation_config: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
    ) -> Optional[str]:
        """
        Apply custom prefix/suffix to context or mark answers in context.

        Args:
            context: Original context text
            augmentation_config: Dict with 'prefix', 'suffix', and optional 'mark_answer' keys
            answer: Answer to find and mark in context (always provided from datasets)

        Returns:
            Augmented context string
        """
        if not context or not augmentation_config:
            return context

        prefix = augmentation_config.get("prefix", "")
        suffix = augmentation_config.get("suffix", "")

        assert (
            prefix and suffix
        ), "Prefix and suffix are required for augmenting context"

        result = context

        assert answer is not None
        assert answer in context

        result = context.replace(answer, f"{prefix}{answer}{suffix}")

        return result

    def __call__(
        self,
        context: Optional[str] = None,
        question: Optional[str] = None,
        user_messages: Union[str, list] = "Please provide your answer.",
        enable_thinking: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
    ) -> str:
        """
        Build the complete prompt for memory benchmark tasks.

        Args:
            context: Long context or background information (optional)
            question: The question to answer
            user_messages: Additional user instructions
            enable_thinking: Whether to enable thinking mode
            augmentation_config: Configuration for context augmentation
            answer: Answer to find and mark in context (for answer marking)

        Returns:
            Formatted prompt string using the model's prompt format
        """
        # Apply context augmentation if configured
        augmented_context = self.augment_context(context, augmentation_config, answer)

        return self.prompt_format.build(
            self.system_prompt(augmented_context, question),
            self.user_messages(user_messages),
            enable_thinking=enable_thinking,
        )


class PasskeyPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for passkey retrieval tasks"""

    system_prompt_format = "You are a helpful AI assistant. Find and return the passkey mentioned in the following text."

    def system_prompt(self, context, question):
        """Create system prompt specifically for passkey tasks"""
        if context:
            return f"{self.system_prompt_format}\n\nText: {context}\n\nQuestion: {question}"
        else:
            return f"{self.system_prompt_format}\n\nQuestion: {question}"

    def augment_context(self, context, augmentation_config, answer):
        if not context or not augmentation_config:
            return context

        prefix = augmentation_config.get("prefix", "")
        suffix = augmentation_config.get("suffix", "")

        assert (
            prefix and suffix
        ), "Prefix and suffix are required for augmenting context"

        result = context

        assert answer is not None
        assert answer in context

        formatted_answer = (
            f"The pass key is {answer}. Remember it. {answer} is the pass key."
        )

        result = context.replace(
            formatted_answer,
            f"{prefix}{formatted_answer}{suffix}",
        )

        return result


class ConversationalQAPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for conversational QA tasks (LoCoMo)"""

    system_prompt_format = "You are a helpful AI assistant. Answer the question based on the conversation history provided."

    def system_prompt(self, context, question):
        """Create system prompt specifically for conversational QA"""
        if context:
            return f"{self.system_prompt_format}\n\nConversation History:\n{context}\n\nQuestion: {question}"
        else:
            return f"{self.system_prompt_format}\n\nQuestion: {question}"


class LongContextQAPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for long context QA tasks (LongBench)"""

    system_prompt_format = "You are a helpful AI assistant. Please read the following document carefully and answer the question based on the information provided."

    def system_prompt(self, context, question):
        """Create system prompt specifically for long context QA"""
        if context:
            return f"{self.system_prompt_format}\n\nDocument:\n{context}\n\nQuestion: {question}, keep the answer short and concise, just return the answer, no other text."
        else:
            return f"{self.system_prompt_format}\n\nQuestion: {question}"


class LongbenchRetrievalPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for long context retrieval tasks (LongBench)"""

    system_prompt_format = "You are a helpful AI assistant. Please read the following document carefully and answer the question based on the information provided."

    def system_prompt(self, context, question):
        """Create system prompt specifically for long context QA"""
        if context:
            return f"{self.system_prompt_format}\n\nDocument:\n{context}\n\nQuestion: Which paragraph talk about the topic of {question}? Just return the paragraph number, like Paragraph 1 , 段落 1, no other text."
        else:
            return f"{self.system_prompt_format}\n\nQuestion: {question}"

    def augment_context(self, context, augmentation_config, answer):
        if not context or not augmentation_config:
            return context

        prefix = augmentation_config.get("prefix", "")
        suffix = augmentation_config.get("suffix", "")

        assert (
            prefix and suffix
        ), "Prefix and suffix are required for augmenting context"

        result = context

        assert answer is not None
        assert answer in context

        match = re.match(r"(.+?)(\d+)$", answer)
        if match:
            answer_prefix = match.group(1)
            answer_num = int(match.group(2))
        else:
            raise ValueError(
                f"Answer {answer} format has no valid paragraph number, the answer should be like 'Paragraph 1' or '段落 1', the given answer is {answer}"
            )

        paragraph_start = context.find(f"{answer_prefix}{answer_num}")
        paragraph_end = context.find(f"{answer_prefix}{answer_num + 1}")
        assert (
            paragraph_start != -1 and paragraph_end != -1
        ), f" {answer_prefix}{answer_num} or {answer_prefix}{answer_num + 1} not found in context"

        answer_paragraph = context[paragraph_start:paragraph_end]

        result = context.replace(
            answer_paragraph, f"{prefix}{answer_paragraph}{suffix}"
        )

        return result


def get_memory_prompt_wrapper(
    task_type: str, prompt_format: PromptFormat
) -> MemoryPromptWrapper:
    """
    Factory function to get appropriate prompt wrapper for memory benchmark task.

    Args:
        task_type: Type of memory task (e.g., 'passkey', 'conversational_qa', 'long_qa')
        prompt_format: PromptFormat instance for the model

    Returns:
        Appropriate MemoryPromptWrapper subclass
    """
    task_type_lower = task_type.lower()

    if "passkey" in task_type_lower:
        return PasskeyPromptWrapper(prompt_format)
    elif any(
        keyword in task_type_lower
        for keyword in ["conversational", "locomo", "conversation"]
    ):
        return ConversationalQAPromptWrapper(prompt_format)
    elif any(keyword in task_type_lower for keyword in ["qa"]):
        return LongContextQAPromptWrapper(prompt_format)
    elif any(keyword in task_type_lower for keyword in ["retrieval", "longbench"]):
        return LongbenchRetrievalPromptWrapper(prompt_format)
    else:
        # Default to general memory prompt wrapper
        return MemoryPromptWrapper(prompt_format)
