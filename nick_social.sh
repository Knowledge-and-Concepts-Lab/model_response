# write a shell script to run this command 2 times. For each time, change the output path to a different file name. like response_1.csv and response_2.csv
# python main.py --exp_name 'q_and_a' --model_type 'gpt-3.5-turbo-0613' --input './experiments/social/prompt.csv' --output './experiments/social/responses/response.csv' --batch_size 256

# for i in {1..100}
# do
#     python main.py --exp_name 'q_and_a' --model_type 'gpt-3.5-turbo-0613' --input './experiments/social/prompt.csv' --output "./experiments/social/responses/response_$i.csv" --batch_size 256
# done

for i in {1..100}
do
    python main.py --exp_name 'q_and_a' --model_type 'gpt-3.5-turbo-0613' --input './experiments/social/prompt.csv' --output "./experiments/social/responses_max_tokens_inf/response_$i.csv" --batch_size 256
done