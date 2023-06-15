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
# python main.py --exp_name 'pairwise' --model_type 'gpt' --model_name 'text-davinci-003' --input './examples/pairwise/prompt.csv' --output './examples/pairwise/response_davinci-003.csv' --batch_size 256
# python main.py --exp_name 'pairwise' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/pairwise/prompt.csv' --output './examples/pairwise/response_flan-t5-xl.csv' --batch_size 256
# python main.py --exp_name 'pairwise' --model_type 'flan' --model_name 'google/flan-t5-xxl' --input './examples/pairwise/prompt.csv' --output './examples/pairwise/response_flan-t5-xxl.csv' --batch_size 256

# test small group of data
# python main.py --exp_name 'pairwise' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/pairwise/prompt_small.csv' --output './examples/pairwise/response_flan-t5-xl.csv' --batch_size 1
# python main.py --exp_name 'triplet' --model_type 'flan' --model_name 'google/flan-t5-xl' --input './examples/triplet/prompt_small.csv' --output './examples/triplet/response_flan-t5-xl-small.csv' --batch_size 1

# pairwise experiments with 20 times of temp 0.7

# for i in $(seq 3 20); do
#     python main.py \
#         --exp_name 'pairwise' \
#         --model_type 'gpt' \
#         --model_name 'text-davinci-003' \
#         --input './examples/pairwise/prompt.csv' \
#         --output "./examples/pairwise/exp_20_temp_0.7/response_davinci-003_${i}.csv" \
#         --batch_size 256 \
#         --temprature 0.7
# done

# GPT-3 expriments
python main.py --exp_name 'pairwise' --model_type 'gpt' --model_name 'text-davinci-002' --input './examples/pairwise/prompt.csv' --output './examples/pairwise/response_davinci-002.csv' --batch_size 256
python main.py --exp_name 'triplet' --model_type 'gpt' --model_name 'text-davinci-002' --input './examples/triplet/prompt.csv' --output './examples/triplet/response_flipped_davinci-002.csv' --batch_size 256
