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

# interactions with transformer
def get_transformer_responses(batches, model, pretrained, batch_size):
    
    responses = []
    prompt_and_response = [] # for saving results
    batches = np.array(list(itertools.chain(*batches)))
    start_time = time.time()
    
    # for gpu computing
    accelerator = Accelerator()
    device = accelerator.device
    
    # prepare the data
    prompt_list = batches.tolist()
    max_length = max([len(prompt.split()) for prompt in prompt_list])
    prompt_dict = {'prompt':prompt_list}
    
    # load the model, for gpu computing
    tokenizer = accelerator.prepare(
        T5Tokenizer.from_pretrained(pretrained)
    )
    
    ds = Dataset.from_dict(prompt_dict)
    ds = ds.map(lambda examples: T5Tokenizer.from_pretrained(pretrained)(examples['prompt'], max_length=max_length, truncation=True, padding='max_length'), batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    flan_model = T5ForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.bfloat16,  cache_dir="./models")
    
    # for gpu computing
    flan_model = flan_model.to(device)
    
    # get the responses
    preds = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = flan_model.generate(input_ids, attention_mask=attention_mask, renormalize_logits = True)
        preds.extend(outputs)
    print('Time taken to generate responses is {}s'.format(time.time()-start_time))
    responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
    del model
    
    # return the results
    for prompt, response in zip(batches, responses):
        prompt_and_response.append([prompt, response])
    return prompt_and_response
    
    
def get_gpt_responses(batches, model, openai_key, temperature):
    openai.api_key = openai_key
    prompt_and_response = []
    for batch in batches:
        for i, prompt in enumerate(batch):
            try:
                completion = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    max_tokens=1024,
                    n=1,
                    temperature=temperature,
                )
            except Exception as e:
                print(e)
                sleep(60)
                completion = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    max_tokens=1024,
                    n=1,
                    temperature=temperature,
                )
            response = completion.choices[0].text.strip()
            # print(response)
            prompt_and_response.append([prompt, response])
    return prompt_and_response
