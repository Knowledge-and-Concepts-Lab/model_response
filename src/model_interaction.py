 #################################################################################
 # The code is originally written by Siddharth Suresh (siddharth.suresh@wisc.edu)#
 # Repurposed and modified by Alex Huang (whuang288@wisc.edu)                    #
 ###################################################################################
import numpy as np
import torch
import itertools
import warnings
import logging
import openai
import time
from datasets import Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import os
from pattern.en import pluralize
from joblib import Parallel, delayed
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle5 as pickle
from accelerate import Accelerator
import inflect
from pathlib import Path
from time import sleep
import pickle5 as pickle
import transformers

def get_responses(batches, model_type):
    batches = batches[0]

    start_time = time.time()
    transformers.logging.set_verbosity_error()

    accelerator = Accelerator()
    device = accelerator.device
    
    if model_type == "llama-7b":
        model = transformers.LlamaForCausalLM.from_pretrained("/mnt/disk-1/llama_hf_7B")
        tokenizer = transformers.LlamaTokenizer.from_pretrained("/mnt/disk-1/llama_hf_7B")
    elif model_type == "alpaca-7b":
        model = transformers.AutoModelForCausalLM.from_pretrained("/mnt/disk-1/alpaca-7b")
        tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/disk-1/alpaca-7b")
    elif model_type == "flan-ul2":
        model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-ul2", torch_dtype=torch.bfloat16, cache_dir="/mnt/disk-1/flan-ul2")         
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-ul2")
    elif model_type == "flan-xxl":
        model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-xxl", torch_dtype=torch.bfloat16, cache_dir="/mnt/disk-1/flan-xxl")         
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-xxl")
    elif model_type == "falcon-7b":
        model = transformers.AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", torch_dtype=torch.bfloat16, cache_dir="/mnt/disk-1/falcon-7b", trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained("tiiuae/falcon-7b")  
        
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    if model_type == "llama-7b" or model_type == "alpaca-7b" or model_type == "falcon-7b":
        # pipeline for text generate
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
        # handle responses
        responses = pipe(batches, max_new_tokens=20, do_sample=False)
        responses = [r[0]['generated_text'] for r in responses]
        responses = [','.join(r.split('\n')[1::]) for r in responses]
    elif model_type == "flan-ul2" or model_type == "flan-xxl":
        # prepare and tokenize inputs
        max_length = max([len(prompt.split()) for prompt in batches])
        inputs = tokenizer(batches, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # generate response
        outputs = model.generate(**inputs, max_length=max_length, do_sample=False)
        # decode outputs
        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    prompt_and_response = list(zip(batches, responses))
 
    print('Time taken to generate responses is {}s'.format(time.time()-start_time))
    return prompt_and_response

# interactions with transformer
def get_transformer_responses(batches, model_type, model_name, batch_size):
    
    responses = []
    prompt_and_response = [] 
    batches = np.array(list(itertools.chain(*batches)))
    start_time = time.time()
    
    # for gpu computing
    accelerator = Accelerator()
    device = accelerator.device
    tokenizer = accelerator.prepare(
        T5Tokenizer.from_pretrained(model_name)
    )
    
    # prepare the dataset
    prompt_list = batches.tolist()
    max_length = max([len(prompt.split()) for prompt in prompt_list])
    prompt_dict = {'prompt':prompt_list}
    
    ds = Dataset.from_dict(prompt_dict)
    ds = ds.map(lambda examples: T5Tokenizer.from_pretrained(model_name)(examples['prompt'], max_length=max_length, truncation=True, padding='max_length'), batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    
    flan_model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16,  cache_dir="./models")
    flan_model = flan_model.to(device)
    
    # get the responses
    preds = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # print(f"input: {input_ids}")
        # print(f"attention_mask: {attention_mask}")
        outputs = flan_model.generate(input_ids, attention_mask=attention_mask, renormalize_logits = True)
        preds.extend(outputs)
    
    responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
    del flan_model
    # return the results
    for prompt, response in zip(batches, responses):
        prompt_and_response.append([prompt, response])
        
    print('Time taken to generate responses is {}s'.format(time.time()-start_time))
    return prompt_and_response
    
    
def get_gpt_responses(batches, model_name, openai_key, temperature, max_tokens):
    # helper function to send a whole to chatgpt
    def send_prompt(batch, prompt_and_response, max_tokens):
        for prompt in batch:
            succeed = False
            completion = None
            while not succeed:
                try:
                    completion = openai.Completion.create(
                        engine = model_name,
                        prompt = prompt,
                        max_tokens = max_tokens,
                        n = 1,
                        # temperature = temperature,
                    )
                    succeed = True
                except Exception as e:
                    print("GPT sleeping...")
                    sleep(60)
            assert completion is not None
            response = completion.choices[0].text.strip().strip(".").strip("\n")
            prompt_and_response.append([prompt, response])

    start_time = time.time()
    openai.api_key = openai_key
    prompt_and_response = []
    
    # can change n_jobs accoding to the size of dataset
    Parallel(n_jobs = 10, require='sharedmem')(delayed(send_prompt)(batch, prompt_and_response, max_tokens) for batch in batches)
    print('Time taken to generate responses is {}s'.format(time.time() - start_time))
    return prompt_and_response
