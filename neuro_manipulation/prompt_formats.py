import abc
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
        
        prompt = f''' {Llama2InstFormat.user_tag} {Llama2InstFormat.system_begin}{system_prompt} {Llama2InstFormat.system_end} {user_messages[0]} {Llama2InstFormat.assistant_tag}'''

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]} {Llama2InstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{Llama2InstFormat.user_tag} {user_messages[mid+1]} {Llama2InstFormat.assistant_tag}"
    
        return prompt

    @staticmethod
    def name_pattern(model_name):
        return 'llama-2' in model_name.lower() and 'chat' in model_name.lower()
    
    
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
        
        prompt = f'''{Llama3InstFormat.begin_of_text}{Llama3InstFormat.system_begin} {system_prompt} {Llama3InstFormat.end_of_turn}{Llama3InstFormat.user_tag}{user_messages[0]}{Llama3InstFormat.end_of_turn}{Llama3InstFormat.assistant_tag}'''
        
        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]}{Llama3InstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{Llama2InstFormat.user_tag}{user_messages[mid+1]}{Llama3InstFormat.end_of_turn}{Llama2InstFormat.assistant_tag}"
    
        return prompt
    
    @staticmethod
    def name_pattern(model_name):
        return 'llama-3' in model_name.lower() and 'instruct' in model_name.lower()


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
        
        prompt = f''' {Llama2InstFormat.user_tag} {system_prompt} {user_messages[0]} {Llama2InstFormat.assistant_tag}'''

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]} {Llama2InstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{Llama2InstFormat.user_tag} {user_messages[mid+1]} {Llama2InstFormat.assistant_tag}"
    
        return prompt

    @staticmethod
    def name_pattern(model_name):
        return 'mistral' in model_name.lower() and 'instruct' in model_name.lower()

     

class PromptFormat:
    format_ls = [Llama2InstFormat, Llama3InstFormat, MistralInstFormat]
    
    @staticmethod
    def get(model_name) -> ModelPromptFormat:
        for format in PromptFormat.format_ls:
            if format.name_pattern(model_name): # more than one pattern may match
                return format
        raise ValueError(f'Prompt format not found for model name: {model_name}. Supported formats: {PromptFormat.format_ls}')
    
    @staticmethod
    def build(model_name, system_prompt, user_messages:list, assistant_messages:list=[] ) -> str:
        return PromptFormat.get(model_name).build(system_prompt, user_messages, assistant_messages)