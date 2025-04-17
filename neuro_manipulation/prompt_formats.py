import abc
import copy

from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class FormatRegistry:
    def __init__(self):
        self.formats = []
    
    def register(self):
        def decorator(cls):
            self.formats.append(cls)
            return cls
        return decorator
    
    def __iter__(self):
        return iter(self.formats)
    
    def __getitem__(self, idx):
        return self.formats[idx]

FORMAT_REGISTRY = FormatRegistry()


class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


def classproperty(func):
    return ClassPropertyDescriptor(func)

class ModelPromptFormat(abc.ABC):
    @classproperty
    @abc.abstractmethod
    def user_tag(cls):
        pass
    
    @classproperty
    @abc.abstractmethod
    def assistant_tag(cls):
        pass
    
    @abc.abstractmethod
    def build(system_prompt, user_messages:list, assistant_answers:list=[]):
        pass
    
    @abc.abstractmethod
    def name_pattern(self, model_name):
        pass

@FORMAT_REGISTRY.register()
class GPT2Format(ModelPromptFormat):
    '''
    for debugging only
    '''
    
    __user_tag = 'User:'
    __assistant_tag = 'Assistant:'
    __end_of_turn = '\n\n'
    
       
    @classproperty
    def user_tag(cls):
        return cls.__user_tag
    
    @classproperty
    def assistant_tag(cls):   
        return cls.__assistant_tag
    
    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn  
   
    @staticmethod
    def build(system_prompt, user_messages:list, assistant_answers:list=[]):
        '''
        User: bbbbbb
        
        Assistant: aaaaa
        
        User: bbbb
        
        Assistant: cccc
        '''
        
        assert len(user_messages), f' user_messages: {user_messages} should not empty'
        assert len(user_messages) - len(assistant_answers) in [0,1], f' user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element'
        
        prompt = f"{GPT2Format.user_tag} {system_prompt if system_prompt else ''} {user_messages[0]}{GPT2Format.end_of_turn}"
        
        for mid in range(len(assistant_answers)):
            prompt += f"{GPT2Format.assistant_tag} {assistant_answers[mid]}{GPT2Format.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{GPT2Format.user_tag} {user_messages[mid+1]}{GPT2Format.end_of_turn}"
    
        return prompt

    @staticmethod
    def name_pattern(model_name):
        return 'gpt2' in model_name.lower()

@FORMAT_REGISTRY.register()
class Llama2InstFormat(ModelPromptFormat):
    __system_begin = '<<SYS>>'
    __system_end = '<</SYS>>'
    __user_tag = '[INST]'
    __assistant_tag = '[/INST]'
    __end_of_turn = '</s>'
    
    @classproperty
    def system_begin(cls):
        return cls.__system_begin
    
    @classproperty
    def system_end(cls):
        return cls.__system_end
    
    @classproperty
    def user_tag(cls):
        return cls.__user_tag
    
    @classproperty
    def assistant_tag(cls):   
        return cls.__assistant_tag

    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn

    @staticmethod
    def build(system_prompt, user_messages:list, assistant_answers:list=[]):
        '''
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
        <s>[INST] {{ user_message_2 }} [/INST]
        '''
        
        assert len(user_messages), f' user_messages: {user_messages} should not empty'
        assert len(user_messages) - len(assistant_answers) in [0,1], f' user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element'
        
        if system_prompt:
            prompt = f''' {Llama2InstFormat.user_tag} {Llama2InstFormat.system_begin}{system_prompt} {Llama2InstFormat.system_end} {user_messages[0]} {Llama2InstFormat.assistant_tag}'''
        else:
            prompt = f''' {Llama2InstFormat.user_tag} {user_messages[0]} {Llama2InstFormat.assistant_tag}'''

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]} {Llama2InstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{Llama2InstFormat.user_tag} {user_messages[mid+1]} {Llama2InstFormat.assistant_tag}"
    
        return prompt

    @staticmethod
    def name_pattern(model_name):
        return 'llama-2' in model_name.lower() and 'chat' in model_name.lower()
    
@FORMAT_REGISTRY.register()
class Llama3InstFormat(ModelPromptFormat):
    __begin_of_text = '<|begin_of_text|>'
    __system_begin = '<|start_header_id|>system<|end_header_id|>'
    __user_tag = '<|start_header_id|>user<|end_header_id|>'
    __assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    __end_of_turn = '<|eot_id|>' 
    
    @classproperty
    def begin_of_text(cls):
        return cls.__begin_of_text
    
    @classproperty
    def system_begin(cls):
        return cls.__system_begin
    
    @classproperty
    def user_tag(cls):
        return cls.__user_tag
    
    @classproperty
    def assistant_tag(cls):   
        return cls.__assistant_tag
    
    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn
    
    @staticmethod
    def build(system_prompt, user_messages:list, assistant_answers:list=[]):
        
        '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        Cutting Knowledge Date: December 2023
        Today Date: 23 July 2024

        You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

        What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        
        assert len(user_messages), f' user_messages: {user_messages} should not empty'
        assert len(user_messages) - len(assistant_answers) in [0,1], f' user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element'
        
        if system_prompt:
            prompt = f'''{Llama3InstFormat.begin_of_text}{Llama3InstFormat.system_begin} {system_prompt} {Llama3InstFormat.end_of_turn}{Llama3InstFormat.user_tag}{user_messages[0]}{Llama3InstFormat.end_of_turn}{Llama3InstFormat.assistant_tag}'''
        else:
            prompt = f'''{Llama3InstFormat.begin_of_text}{Llama3InstFormat.user_tag}{user_messages[0]}{Llama3InstFormat.end_of_turn}{Llama3InstFormat.assistant_tag}'''
        
        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]}{Llama3InstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{Llama3InstFormat.user_tag}{user_messages[mid+1]}{Llama3InstFormat.end_of_turn}{Llama3InstFormat.assistant_tag}"
    
        return prompt
    
    @staticmethod
    def name_pattern(model_name):
        return 'llama-3' in model_name.lower() and 'instruct' in model_name.lower()


@FORMAT_REGISTRY.register()
class MistralInstFormat(ModelPromptFormat):
    __user_tag = '[INST]'
    __assistant_tag = '[/INST]'
    __end_of_turn = '</s>'
    @classproperty
    def user_tag(cls):
        return cls.__user_tag
    
    @classproperty
    def assistant_tag(cls):   
        return cls.__assistant_tag
    
    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn 
    
    @staticmethod
    def build(system_prompt, user_messages:list, assistant_answers:list=[]):
        '''
        <s> [INST] bbbbbb [/INST] aaaaa</s> [INST] bbbb [/INST] cccc</s>
        '''
        
        assert len(user_messages), f' user_messages: {user_messages} should not empty'
        assert len(user_messages) - len(assistant_answers) in [0,1], f' user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element'
        
        if system_prompt:
            prompt = f''' {MistralInstFormat.user_tag} {system_prompt} {user_messages[0]} {MistralInstFormat.assistant_tag}'''
        else:
            prompt = f''' {MistralInstFormat.user_tag} {user_messages[0]} {MistralInstFormat.assistant_tag}'''

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]} {MistralInstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{MistralInstFormat.user_tag} {user_messages[mid+1]} {MistralInstFormat.assistant_tag}"
    
        return prompt

    @staticmethod
    def name_pattern(model_name):
        return 'mistral' in model_name.lower() and 'instruct' in model_name.lower()

@FORMAT_REGISTRY.register()
class RWKVsFormat(ModelPromptFormat):
    __user_tag = 'User:'
    __assistant_tag = 'Assistant:'
    __end_of_turn = '\n\n'
    
    @classproperty
    def user_tag(cls):
        return cls.__user_tag
    
    @classproperty
    def assistant_tag(cls):   
        return cls.__assistant_tag
    
    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn  
   
    @staticmethod
    def build(system_prompt, user_messages:list, assistant_answers:list=[]):
        '''
        User: bbbbbb
        
        Assistant: aaaaa
        
        User: bbbb
        
        Assistant: cccc
        '''
        
        assert len(user_messages), f' user_messages: {user_messages} should not empty'
        assert len(user_messages) - len(assistant_answers) in [0,1], f' user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element'
        
        prompt = f"{RWKVsFormat.user_tag} {system_prompt if system_prompt else ''} {user_messages[0]}{RWKVsFormat.end_of_turn}"
        
        for mid in range(len(assistant_answers)):
            prompt += f"{RWKVsFormat.assistant_tag} {assistant_answers[mid]}{RWKVsFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{RWKVsFormat.user_tag} {user_messages[mid+1]}{RWKVsFormat.end_of_turn}"
    
        return prompt

    @staticmethod
    def name_pattern(model_name):
        return 'rwkv' in model_name.lower()

class ManualPromptFormat:
    ''' 
    Manually defined prompt format.
    '''
    
    @staticmethod
    def get(model_name) -> ModelPromptFormat:
        for format in FORMAT_REGISTRY:
            if format.name_pattern(model_name): # more than one pattern may match
                return format
        raise ValueError(f'Prompt format not found for model name: {model_name}. Supported formats: {list(FORMAT_REGISTRY)}')
    
    @staticmethod
    def build(model_name, system_prompt, user_messages:list, assistant_messages:list=[] ) -> str:
        return ManualPromptFormat.get(model_name).build(system_prompt, user_messages, assistant_messages)
    
class PromptFormat:
    '''
    Format prompts using the tokenizer's chat template.

    This class uses the official chat template defined by the model providers,
    which ensures compatibility with the expected format for each model.
    
    In case the chat template is not available or fails, it falls back to the manual format definitions.
    '''

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.model_name = tokenizer.name_or_path
    
    def build(self, system_prompt, user_messages:list, assistant_messages:list=[] ) -> str:
        """
        Build a prompt string using the tokenizer's chat template
        
        Args:
            model_name (str): Name of the model (used for fallback)
            system_prompt (str): System prompt to use
            user_messages (list): List of user messages
            assistant_messages (list): List of assistant messages (optional)
            
        Returns:
            str: The formatted prompt string
        """
        chat = []
        
        # Only add system prompt if it's provided
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
            
        assert len(user_messages), f' user_messages: {user_messages} should not empty'
        for user_message, assistant_message in zip(user_messages, assistant_messages):
            chat.append({"role": "user", "content": user_message})
            chat.append({"role": "assistant", "content": assistant_message})
        
        # If there are still user messages left (odd number of total messages), add the last one
        if len(user_messages) > len(assistant_messages):
            chat.append({"role": "user", "content": user_messages[-1]})

        try:
            prompt_str = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            # Verify messages appear in correct order in the prompt string
            self._verify_message_order_in_prompt(prompt_str, chat)
            return prompt_str
        except Exception as e:
            logger.warning(f"Error applying chat template for model {self.model_name}: {e}")
            # In case the error from system prompt, try to merge system prompt to the first user message
            if system_prompt and len(chat) > 1:
                logger.info("Try to merge system prompt into first message")
                raw_chat_1 = copy.deepcopy(chat[1]["content"])
                for format in FORMAT_REGISTRY:
                    if format.name_pattern(self.model_name):
                        try:
                            if chat[0]["role"] == "system":
                                chat.pop(0) # remove system prompt
                                chat[0]["content"] = f'{format.system_begin}{system_prompt}{format.system_end}\n\n{raw_chat_1}'
                            prompt_str = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                            # Verify messages appear in correct order in the prompt string
                            self._verify_message_order_in_prompt(prompt_str, chat)
                            return prompt_str
                        except Exception as e:
                            logger.warning(f"Error applying chat template for model {self.model_name}: {e}")
                            continue
            
            # Fallback to ManualPromptFormat if all else fails
            logger.info(f"Falling back to ManualPromptFormat")
            prompt_str = ManualPromptFormat.build(self.model_name, system_prompt, user_messages, assistant_messages)
            return prompt_str

    def _verify_message_order_in_prompt(self, prompt_str, chat):
        """
        Verify that message contents appear in the correct order in the final prompt string.
        
        Args:
            prompt_str (str): The formatted prompt string
            chat (list): The list of chat messages
            
        Raises:
            AssertionError: If message content order is not preserved
        """
        previous_idx = -1
        
        # Check each message content appears in the expected order
        for msg in chat:
            content = msg["content"]
            if content: 
                # Check if the content appears in the prompt string
                if content not in prompt_str:
                    raise AssertionError(f"Message content '{content[:20]}...' not found in prompt string")
                
                # Find the start index of this content and ensure it comes after the previous content
                current_idx = prompt_str.find(content)
                if current_idx <= previous_idx and previous_idx != -1:
                    raise AssertionError(f"Message order not preserved. Content '{content[:20]}...' appears out of order in prompt string")
                
                previous_idx = current_idx

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt_format = PromptFormat(tokenizer)
    from_prompt_format = prompt_format.build("You are a helpful assistant", ["Hello, how are you?"], ["I'm doing well, thank you!"])
    from_tk_template = tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}], tokenize=False, add_generation_prompt=True)
    print(from_prompt_format)
    print(from_tk_template)
    print(from_prompt_format == from_tk_template)