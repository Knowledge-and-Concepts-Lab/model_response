# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 128

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 128
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 256

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256

# python main.py --exp_name 'triplet' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256

# python main.py --exp_name 'feature_and_concept' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/feature_and_concept/features.csv' './examples/feature_and_concept/animals.csv' --output './examples/feature_and_concept/response.csv' --batch_size 1

# pairwise experiments
python main.py --exp_name 'pairwise' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/pairwise/prompt.csv' --output './examples/pairwise/response_davinci-003.csv' --batch_size 256