import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

LLM_PATH = "/workspace/patentdoc-copilot/models/Qwen2.5-7B-Instruct"
SYSTEM_MSG = "You are a helpful assistant."


@st.cache_resource
def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_PATH,
        local_files_only=True
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        local_files_only=True,
        quantization_config=bnb,
        device_map="cuda"  # keep on GPU
    )

    model.eval()
    return tokenizer, model


def llm_generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.85,
    repeat_penalty: float = 1.2,
    stop_strings=None
):
    if stop_strings is None:
        stop_strings = ["\n\n"]

    tokenizer, model = get_llm()

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        try:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
                stop_strings=stop_strings,
                tokenizer=tokenizer,
            )
        except TypeError:
            # Fallback for older Transformers
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
            )

    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
