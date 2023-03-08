# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 128

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 128
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 256

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256

# python main.py --exp_name 'triplet' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/triplet/prompt.csv' --output './examples/triplet/response.csv' --batch_size 256
python main.py --exp_name 'q_and_a' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/q_and_a/prompt.csv' --output './examples/q_and_a/response.csv' --batch_size 256
