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

# interactions with transformer
def get_transformer_responses(batches, model, pretrained, exp_name, batch_size):
    
    prompt_and_response = [] # for saving results
    start_time = time.time()
    
    batches = np.array(list(itertools.chain(*batches)))
    
    if model == 'flan':
        responses = []
        
        if exp_name == 'triplet':
            start_time = time.time()
            
            # for gpu computing
            accelerator = Accelerator()
            device = accelerator.device
            
            # single out each part of the prompt
            anchor = batches[:,0]
            concept1 = batches[:,1]
            concept2 = batches[:,2]
            prompts = batches[:,3]
            tokens = batches[:,4]
            
            # prepare the data
            prompt_list = prompts.tolist()
            max_length = max([len(prompt.split()) for prompt in prompt_list])
            prompt_dict = {'prompt': prompt_list}

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
           
            # generate responses
            preds = []
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = flan_model.generate(input_ids, attention_mask=attention_mask, renormalize_logits = True)
                preds.extend(outputs)
            print('Time taken to generate responses is {}s'.format(time.time()-start_time))
            responses = tokenizer.batch_decode(preds, skip_special_tokens=True)
            
            # return the results
            del flan_model
            for anchor, concept1, concept2, response , prompt in zip(anchor, concept1, concept2, responses, prompts):
                prompt_and_response.append([prompt, response])
                # prompt_and_response.append([anchor, concept1, concept2, response, tokens, prompt])
            return prompt_and_response
            
        elif exp_name == 'q_and_a':
            
            start_time = time.time()
            
            # for gpu computing
            accelerator = Accelerator()
            device = accelerator.device
            
            # prepare the data
            prompts = batches
            prompt_list = prompts.tolist()
            max_length = max([len(prompt.split()) for prompt in prompt_list])
            prompt_dict = {'prompt':batches.tolist()}
            
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
            for prompt, response in zip(prompts, responses):
                prompt_and_response.append([prompt, response])
            return prompt_and_response

        else:
            logging.error('Undefined experiment. Only feature listing and triplet implemented')
    else:
        logging.error('Undefined model. Only flan implemented')
    
    
