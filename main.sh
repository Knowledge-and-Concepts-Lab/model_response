# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_256.csv' --batch_size 256
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_128.csv' --batch_size 128

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_128.csv' --batch_size 128
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response_256.csv' --batch_size 256

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './samples/q_and_a_sample_prompt.csv' --output './samples/q_and_a_sample_response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './samples/q_and_a_sample_prompt.csv' --output './samples/q_and_a_sample_response.csv' --batch_size 256

# python main.py --exp_name 'triplet' --model_type 'gpt' --model_name 'text-davinci-003' --input './samples/triplet_sample_prompt.csv' --output './samples/triplet_sample_response.csv' --batch_size 256
# python main.py --exp_name 'q_and_a' --model_type 'gpt' --model_name 'text-davinci-003' --input './samples/q_and_a_sample_prompt.csv' --output './samples/q_and_a_sample_response.csv' --batch_size 256
python main.py --exp_name 'feature_and_concept' --model_type 'gpt' --model_name 'text-davinci-003' --input './samples/feature_and_concept_sample_prompt.csv' --output './samples/feature_and_concept_sample_response.csv' --batch_size 256