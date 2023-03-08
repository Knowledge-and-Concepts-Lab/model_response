#  Get Responses from LLMs

This repository contains the code to get responses from Large language models online.

## Requirements

To install conda on your remote Linux server, use the following commands:

```sh
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

To set up the environment with conda, use the following commands:

```sh
conda create --name get_responses -c conda-forge python=3.7 pattern
conda activate get_responses
python -m pip install -r requirements.txt
```

## Run the scripts

#### Example Run:

```sh
python main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256
```

```sh
python main.py --exp_name 'q_and_a' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256
```

### Changing parameters

- `exp_name`: Your experiment type

    - q_and_a: The most generic type of experiment. Simply provide a csv file that has prompts on each line.  

    - triplet: Provide concepts A, B, and C on each line. Let the LLM decide if concept A is closer to concept B or concept C.

    - feature_and_concept: Provide two csv files. One with features on each line and one with concepts on each line. Let the LLM generate a prompt for each (feature, concept) pair.

- `model_type`: there is only `flan` and `gpt` available for now

- `model_name`: google/flan-t5-xl, google/flan-t5-xxl, etc for FLAN and text-davinci-002, text-davinci-003, etc for GPT

- `input`: The path to your input csv file(s)

- `output`: The desired path to store your output csv file

- `batch_size`: The batch size of data that is fed to the LLM

<br>

## Setting up the API key 

To use the gpt model, please create a file called "api_key.txt" under the main directory, and paste your openai API key in it. Please refer to this <a href = "https://www.educative.io/courses/open-ai-api-natural-language-processing-python/7DxorX8xA0O"> website </a> for more information. Note that depending on your subscription plan, your rate limit will be different and the time it takes to get all your responses may varies according to it.

## Prompt Engineering

To play around with prompt engineering, check out `src/prompt_engineering.ipynb`.