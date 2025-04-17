from transformers.pipelines import TextGenerationPipeline
from .rep_control_reading_vec import WrappedReadingVecModel

class RepControlPipelineWrappedBlock(TextGenerationPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 **kwargs):
        
        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert block_name == "decoder_block" or "LlamaForCausalLM" in model.config.architectures, f"{model.config.architectures} {block_name} not supported yet"
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        self.block_name = block_name
        self.layers = layers

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        
        # Make sure tokenizer has a pad token
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "eos_token_id"):
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
                print(f"Setting pad_token_id to eos_token_id: {self.tokenizer.pad_token_id}")
            elif self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Setting pad_token to eos_token: {self.tokenizer.pad_token}")
   
    def __call__(self, text_inputs, reset_hooks=True, activations=None, token_pos=None, masks=None, normalize=False, operator='linear_comb', **kwargs):
        # Handle custom parameters
        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)

        # Only pass the standard parameters to the parent class
        outputs = super().__call__(text_inputs, **kwargs)
        
        # Reset hooks if requested
        if reset_hooks:
            self.wrapped_model.reset()

        return outputs