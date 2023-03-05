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
python main.py --exp_name 'q_and_a' --model 'flan' --pretrained 'google/flan-t5-xl' --input './samples/q_and_a_sample_prompt.csv' --output './samples/q_and_a_sample_response.csv' --batch_size 256
```
### Changing parameters

- `exp_name`: Your experiment type

    - q_and_a: The most generic type of experiment. Simply provide a csv file that has prompts on each line.  

    - triplet: Provdie concepts A, B, and C on each line. Let the LLM decide if concept A is closer to concept B or concept C.

- `model`: there is only `flan` and `gpt` available for now

- `pretrained`: google/flan-t5-xl, google/flan-t5-xxl (this is specific to FLAN model)

- `input`: The path to your input csv file

- `output`: The desired path to store your output csv file

- `batch_size`: The batch size of data that is fed to the LLM

<br>

## Setting up the API key 

To use the gpt model, please create a file called "api_key.txt" under the main directory, and paste your openai API key in it. Please refer to this <a href = "https://www.educative.io/courses/open-ai-api-natural-language-processing-python/7DxorX8xA0O"> website </a> for more information. Note that depending on your subscription plan, your rate limit will be different and may take longer to get all your responses.  

## Prompt Engineering

To play around with prompt engineering, check out `src/prompt_engineering.ipynb`.