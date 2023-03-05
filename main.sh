# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model 'flan' --pretrained 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_256.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model 'flan' --pretrained 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_128.csv' --batch_size 128

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model 'flan' --pretrained 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_128.csv' --batch_size 128
# python main.py --exp_name 'triplet' --model 'flan' --pretrained 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_256.csv' --batch_size 256

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'q_and_a' --model 'flan' --pretrained 'google/flan-t5-xxl' --input './samples/q_and_a_sample_prompt.csv' --output './samples/q_and_a_sample_response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model 'flan' --pretrained 'google/flan-t5-xl' --input './samples/q_and_a_sample_prompt.csv' --output './samples/q_and_a_sample_response.csv' --batch_size 256

# python main.py --exp_name 'triplet' --model 'gpt' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_256.csv' --batch_size 256
python main.py --exp_name 'q_and_a' --model 'gpt' --input './samples/q_and_a_sample_prompt.csv' --output './samples/q_and_a_sample_response.csv' --batch_size 256