export CUDA_VISIBLE_DEVICES=3

export TASK_NAME=mrpc # --> 230 steps per epoch with bsz=16 (number of steps depends on training set size and batch size)

# following the recommendations of https://arxiv.org/abs/2006.04884 we use
# - AdamW with bias correction
# - train for 20 epochs to increase the number of steps
# - use learning rate warmup during the first 10% of the total steps
# - repeat fine-tuning using different random seeds
# - don't use early stopping


python run_text_classification.py \
  --model_name_or_path bert-large-uncased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 20 \
  --num_warmup_steps 460 \
  --output_dir /logfiles/$TASK_NAME/ \
  --num_repetitions 25
