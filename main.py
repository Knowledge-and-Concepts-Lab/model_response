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

# the helper function for saving the responses
def save_responses(reponses, file_path):
    with open(file_path, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write = csv.writer(f)
        write.writerows(reponses)
    
# the helper function for running the experiment
def run_exp(exp_name,  
            model_type,
            model_name, 
            input_path,
            output_path, 
            batch_size, 
            max_tokens = 256,
            openai_api_key = None 
            ):
    
    # get the batches accodring to the experiment type
    if exp_name == 'triplet':
        assert len(input_path) == 1, "Triplet Experiment should only have one input file"
        input_file = np.loadtxt(input_path[0], delimiter=',', dtype = str)
        batches = make_prompt_batches_triplet(input_file)
    elif exp_name == 'q_and_a':
        assert len(input_path) == 1, "Q&A Experiment should only have one input file"
        input_file = np.loadtxt(input_path[0], delimiter=',', dtype = str)
        batches = make_prompt_batches_q_and_a(input_file)
    elif exp_name == 'feature_and_concept':
        assert len(input_path) == 2, "Feature and Concept Experiment should have two input files"
        feature_file = np.loadtxt(input_path[0], delimiter='\n', dtype = str)
        batches = make_prompt_batches_feature(feature_file)
    elif exp_name == 'pairwise':
        assert len(input_path) == 1, "Pairwise Experiment should only have input files"
        feature_file = np.loadtxt(input_path[0], delimiter=',', dtype = str)
        batches = make_prompt_batches_pairwise(exp_name, feature_file)
    else:
        logging.error('Undefined task. Only feature listing and triplet implemented')
    
    # print out info about this run
    print('Running experiment {} on data {} using {} model. Please wait for it to finish'.format(exp_name, input_path, model_type))
    
    # get and save the responses
    if model_type == 'flan':
        responses = get_transformer_responses(batches, model_type, model_name, batch_size)
    elif model_type == 'gpt':
        openai_key = Path(f"api_key").read_text()
        responses = get_gpt_responses(batches, model_name, openai_key, 0, max_tokens)
    else:
        logging.error('Only flan and gpt implemented now.')
    
    
    # pipeline specific for the Feature and Concept experiment
    if exp_name == "feature_and_concept":
        with open( output_path, 'w') as output_file:
            with open(input_path[1], 'r') as concept_file:
                concepts = concept_file.readlines()
                for concept in concepts:
                    concept = concept.strip("\n")
                    for _, prompt in responses:
                        output_file.write(prompt.replace("[placeholder]", concept) + "\n")
    else:
        save_responses(responses, output_path)
    return 

# the main method, handling command line arguments
def main():
    
    # parse the arguments
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--exp_name', default = None,
                    type=str, help=""" the experiment type that you are doing""")
    parser.add_argument('--model_type', default = None,
                    type=str, help="""flan or gpt""")
    parser.add_argument('--model_name', default = None,
                    type=str, help = """ the specific model name you are using""")
    parser.add_argument('--input', default=[], nargs='*',
                        help="""path to the input file""")
    parser.add_argument('--output', type=str, default = None,
                        help="""path to the output file""")
    parser.add_argument('--batch_size', type = int, default=256, 
                    help = """The batch size of data that is fed to the LLM""")
    args = parser.parse_args()
    
    # check if arguments was provided
    assert args.exp_name != None
    assert args.model_type != None
    assert args.model_name != None
    assert len(args.input) != 0
    assert args.output != None
    
    
    # log the info to the log file
    os.makedirs("logs/", exist_ok=True)
    logging.basicConfig(filename="logs/{}_{}.log".format(args.exp_name, args.model_type), level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.warning('is when this event was logged.')
    logging.info('Running experiments with the following parameters')
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)
    
    # call the helper function to do the actual work
    run_exp(exp_name = args.exp_name,  
            model_type = args.model_type, 
            model_name = args.model_name,
            input_path = args.input,
            output_path = args.output, 
            batch_size = args.batch_size,
            max_tokens = 256,
            openai_api_key = None, 
           )        

if __name__=="__main__":
    main()
