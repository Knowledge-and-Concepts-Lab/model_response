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
        T5Tokenizer.from_pretrained()
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
        outputs = flan_model.generate(input_ids, attention_mask=attention_mask, renormalize_logits = True)
        preds.extend(outputs)
    
    responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
    del model
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
                        temperature = temperature,
                    )
                    succeed = True
                except Exception as e:
                    print("GPT sleeping...")
                    sleep(60)
            assert completion is not None
            response = completion.choices[0].text.strip().strip(".")
            prompt_and_response.append([prompt, response])

    start_time = time.time()
    openai.api_key = openai_key
    prompt_and_response = []
    
    # can change n_jobs accoding to the size of dataset
    Parallel(n_jobs = 10, require='sharedmem')(delayed(send_prompt)(batch, prompt_and_response, max_tokens) for batch in batches)
    print('Time taken to generate responses is {}s'.format(time.time() - start_time))
    return prompt_and_response
