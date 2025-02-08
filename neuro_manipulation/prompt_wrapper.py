from pydantic import BaseModel
from neuro_manipulation.prompt_formats import PromptFormat


class PromptWrapper:
    system_prompt_format = 'You are an average American.' 
    
    def __init__(self, prompt_format: PromptFormat):
        self.prompt_format = prompt_format
        
    def system_prompt(self, event, options):
        return self.system_prompt_format.format(event=event) + "\n" + "\n".join([f" Option {i+1}. {option}" if not option.startswith('Option') else option for i, option in enumerate(options)])
    
    def user_messages(self, user_messages):
        if type(user_messages) == str:
            return [user_messages]
        return user_messages
    
    def __call__(self, event, options, user_messages):
        return self.prompt_format.build(self.system_prompt(event, options), self.user_messages(user_messages))
        
class ReactionPromptWrapper(PromptWrapper):
    system_prompt_format = "You are an average American. {event} What will you do?"
    
        
class GameReactPromptWrapper(PromptWrapper):
    system_prompt_format = "You are Alice, an average American. You are in the following situation: {event}. "
    
    def __init__(self, prompt_format: PromptFormat, response_format: BaseModel):
        super().__init__(prompt_format)
        self.response_format = response_format
        assert hasattr(response_format, 'example') and callable(getattr(response_format, 'example')), f"response_format should have an example method"
        
    def system_prompt(self, event, options):
        return self.system_prompt_format.format(event=event) + "\n" + \
    "\n".join([f" Option {i+1}. {option}" if not option.startswith('Option') else option for i, option in enumerate(options)]) + \
    "\n" + self.format_instruction()
    
    def format_instruction(self):
        return f"response in json format, with the following structure: {self.response_format.example()}"