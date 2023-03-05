 #################################################################################
 # The code is originally written by Siddharth Suresh (siddharth.suresh@wisc.edu)#
 # Repurposed and modified by Alex Huang (whuang288@wisc.edu)                    #
 #################################################################################


import argparse, os
import logging
import csv
import pandas as pd
import numpy as np
from src.model_interaction import *
from src.prompt_generation import *

DEFAULT_DIR = os.path.abspath(os.getcwd())
DATASET_DIR = os.path.join(DEFAULT_DIR, "data/") 
DEFAULT_RESULTS_DIR = os.path.join(DEFAULT_DIR, "results/")

# the helper function for saving the responses
def save_responses(reponses, file_path):
    with open(file_path, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write = csv.writer(f)
        write.writerows(reponses)
    
# the helper function for running the experiment
def run_exp(exp_name,  
            model,
            pretrained, 
            input_path,
            output_path, 
            batch_size, 
            openai_api_key = None 
            ):
    
    prompts = np.loadtxt(input_path, delimiter=',', dtype = str)
    
    # get the batches accodring to the experiment type
    if exp_name == 'triplet':
        batches = make_prompt_batches_triplet(prompts)
    elif exp_name == 'q_and_a':
        batches = make_prompt_batches_q_and_a(prompts)
    else:
        logging.error('Undefined task. Only feature listing and triplet implemented')
    
    # print out info about this run
    print('Running experiment {} on data {} using {} model. Please wait for it to finish'.format(exp_name, input, model))
    
    # get and save the responses
    if model == 'flan':
        responses = get_transformer_responses(batches, model, pretrained, exp_name,  batch_size)
    elif model == 'gpt':
        openai_key = Path(f"api_key").read_text()
        responses = get_gpt_responses(batches, "text-davinci-003", openai_key, 0)
    else:
        logging.error('Only flan implemented now.')
    save_responses(responses, output_path)
    return 

# the main method, handling command line arguments
def main():
    
    # parse the arguments
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--exp_name', default = None,
                    type=str, help=""" The type of the experiment you are doing""")
    parser.add_argument('--model', default = None,
                    type=str, help=""" Name of the feature listing file""")
    parser.add_argument('--pretrained', default = None,
                    type=str, help = """The dataset that the model is pretrained on""")
    parser.add_argument('--input', type=str, default = None,
                        help="""path to the input file""")
    parser.add_argument('--output', type=str, default = None,
                        help="""path to the output file""")
    parser.add_argument('--batch_size', type = int, default=256, 
                    help = """The batch size of data that is fed to the LLM""")
    args = parser.parse_args()
    
    # check if arguments was provided
    assert args.exp_name is not None
    assert args.model is not None
    assert args.input is not None
    assert args.output is not None
    if args.model == "flan":
        assert args.pretrained is not None
    
    # log the info to the log file
    os.makedirs("logs/", exist_ok=True)
    logging.basicConfig(filename="logs/{}_{}.log".format(args.exp_name, args.model), level=logging.DEBUG, # encoding='utf-8',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.warning('is when this event was logged.')
    logging.info('Running experiments with the following parameters')
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    # call the helper function to do the actual work
    run_exp(exp_name = args.exp_name,  
            model = args.model, 
            pretrained = args.pretrained,
            input_path = args.input,
            output_path = args.output, 
            batch_size = args.batch_size,
            openai_api_key = None, 
           )

if __name__=="__main__":
    main()
