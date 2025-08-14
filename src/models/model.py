#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, PreTrainedTokenizerFast
from anchor import checkpoints_root
import torch


CONTEXT_CHUNK_TOKEN_START = "[CONTEXT]"
CONTEXT_CHUNK_TOKEN_END = "[/CONTEXT]"
INST_TOKEN_START = "[INST]"
INST_TOKEN_END = "[/INST]"


def build_tokenizer(model_hf_signature, model_name, hf_access_token):


    try:
        tokenizer = AutoTokenizer.from_pretrained(model_hf_signature, cache_dir=str(checkpoints_root), trust_remote_code=True, token=hf_access_token)
    except:
        print("Cannot load HF model/tokenizer for given signature...Probably because you use a local path.")
        print("Retrying loading with model name...")
        if model_name == "mistral":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", cache_dir=str(checkpoints_root), trust_remote_code=True, token=hf_access_token)
        elif model_name == "qwen":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=str(checkpoints_root), trust_remote_code=True, token=hf_access_token)
        elif model_name == "phi":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-small-128k-instruct", cache_dir=str(checkpoints_root), trust_remote_code=True, token=hf_access_token)
        elif model_name == "llama":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", cache_dir=str(checkpoints_root), trust_remote_code=True, token=hf_access_token)
        else:
            raise NotImplementedError
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "e2e" in model_hf_signature:
        # Add special tokens for e2e mode
        special_tokens_dict = {'additional_special_tokens': [CONTEXT_CHUNK_TOKEN_START, CONTEXT_CHUNK_TOKEN_END]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Tokenizer has added {num_added_toks} additional tokens...")

    return tokenizer


def build_model(model_hf_signature, model_name, hf_access_token, args):
    
    if "e2e" in model_hf_signature and "token" not in model_hf_signature:
        # inference with our method
        from train.icr2_train.modeling_mistral_icr2 import MistralForCausalLM, MistralForRetrieveLM
        
        ret_model = MistralForRetrieveLM.from_pretrained(
            model_hf_signature,
            token=hf_access_token,
            cache_dir=str(checkpoints_root),
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        gen_model = MistralForCausalLM.from_pretrained(
            model_hf_signature,
            token=hf_access_token,
            cache_dir=str(checkpoints_root),
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        print("Retrieval model:", ret_model)
        print("Generate model:", gen_model)

        model = (ret_model, gen_model)

    elif "e2e" in model_hf_signature and "token" in model_hf_signature:
        # inference with our method
        from train.icr2_train.modeling_mistral_icr2_token import MistralForCausalLM, MistralForRetrieveLM
        
        ret_model = MistralForRetrieveLM.from_pretrained(
            model_hf_signature,
            token=hf_access_token,
            cache_dir=str(checkpoints_root),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",

        )

        gen_model = MistralForCausalLM.from_pretrained(
            model_hf_signature,
            token=hf_access_token,
            cache_dir=str(checkpoints_root),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        model = (ret_model, gen_model)

    elif args.probe_attention:
        from train.icr2_train.modeling_mistral import MistralForCausalLM

        print("Using mistral model with probing attention.")

        model = MistralForCausalLM.from_pretrained(
                    model_hf_signature,
                    torch_dtype="auto",
                    device_map='auto',
                    use_flash_attention_2="flash_attention_2",
                    trust_remote_code=True,
                )
        
        print(model)
        
        if hasattr(model, 'hf_device_map'):
            print("Device map:", model.hf_device_map)
        else:
            print("No device map found. The model might not be using model parallelism.")

    elif  args.block_rta_context_decoding or args.hierarchical_mask_decoding:
        from transformers import MistralForCausalLM
        model = MistralForCausalLM.from_pretrained(
                    model_hf_signature,
                    torch_dtype="auto",
                    device_map='auto',
                    attn_implementation="sdpa", # "flash_attention_2"
                    trust_remote_code=True,
                )

    elif model_hf_signature == "Qwen/Qwen2-7B-Instruct" or model_hf_signature == "microsoft/Phi-3-small-128k-instruct" or model_hf_signature == "meta-llama/Llama-3.1-8B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_hf_signature,
            token=hf_access_token,
            cache_dir=str(checkpoints_root),
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            load_in_4bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_hf_signature,
            token=hf_access_token,
            cache_dir=str(checkpoints_root),
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
        )

    if isinstance(model, tuple):
        for m in model:
            m.eval()
    else:
        model.eval()
        print(model.dtype)
    
    return model
