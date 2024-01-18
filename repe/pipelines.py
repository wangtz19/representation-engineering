from transformers import AutoModel, AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY
from .rep_reading_pipeline import RepReadingPipeline
from .rep_control_pipeline import RepControlPipeline
from functools import partial

def repe_pipeline_registry():
    PIPELINE_REGISTRY.register_pipeline(
        "rep-reading",
        pipeline_class=RepReadingPipeline,
        pt_model=AutoModel,
    )

    PIPELINE_REGISTRY.register_pipeline(
        "rep-control",
        pipeline_class=RepControlPipeline,
        pt_model=AutoModelForCausalLM,
    )

    PIPELINE_REGISTRY.register_pipeline(
        "rep-control-qwen",
        pipeline_class=partial(RepControlPipeline, model_name="qwen"),
        pt_model=AutoModelForCausalLM,
    )

