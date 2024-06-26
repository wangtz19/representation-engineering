from transformers.pipelines import TextGenerationPipeline
# from .rep_control_reading_vec import WrappedReadingVecModel

class RepControlPipeline(TextGenerationPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 generation_config=None,
                 model_name="llama",
                 **kwargs):
        
        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert block_name == "decoder_block" or "LlamaForCausalLM" in model.config.architectures, f"{model.config.architectures} {block_name} not supported yet"
        if model_name == "llama":
            from .rep_control_reading_vec import WrappedReadingVecModel
        elif model_name == "qwen":
            from .rep_control_reading_vec_qwen import WrappedReadingVecModel
        else:
            raise NotImplementedError(f"{model_name} not supported yet")
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        self.block_name = block_name
        self.layers = layers

        if generation_config is not None:
            kwargs["generation_config"] = generation_config
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
   
    def __call__(self, text_inputs, activations=None, **kwargs):

        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)

        outputs = super().__call__(text_inputs, **kwargs)
        self.wrapped_model.reset()

        return outputs