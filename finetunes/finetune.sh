formatted_time=$(date +"%Y-%m-%d-%H-%M-%S")
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export HF_ENDPOINT=https://hf-mirror.com

# Get the task name from arguments
TASK=$1

function finetune_protein_model() {
  local TRAIN_FILE=$1
  local VALID_FILE=$2
  local TEST_FILE=$3
  local PROBLEM_TYPE=$4
  local NUM_LABELS=$5
  local TASK_NAME=$6
  local CROSS_FOLD=${7:-'0'}
  local CROSS_FOLD_INDEX=${8:-'0'}
  local formatted_time=$(date +%Y%m%d_%H%M%S)

  python finetunes/finetune.py \
    --log_gradient_norm="False" \
    --config_path="/home/chengjiabei/ProLM_ckpt" \
    --tokenizer_path="AI4Protein/prolm_8M" \
    --train_file="$TRAIN_FILE" \
    --valid_file="$VALID_FILE" \
    --test_file="$TEST_FILE" \
    --train_batch_size="4" \
    --eval_batch_size="4" \
    --num_workers="0" \
    --seed="42" \
    --max_steps="10000" \
    --max_epochs="-1" \
    --accumulate_grad_batches="8" \
    --lr="0.0001" \
    --adam_beta1="0.9" \
    --adam_beta2="0.999" \
    --adam_epsilon="1e-7" \
    --gradient_clip_value="100.0" \
    --gradient_clip_algorithm="norm" \
    --precision="bf16-mixed" \
    --weight_decay="0.0" \
    --scheduler_type="linear" \
    --check_val_every_n_epoch="1" \
    --val_check_interval="0.75" \
    --save_model_dir="finetune_checkpoint" \
    --save_model_name="prolm_8M_$formatted_time" \
    --log_steps="5" \
    --logger="tensorboard" \
    --logger_project="prolm" \
    --logger_run_name="prolm_8M_$TASK_NAME_$formatted_time" \
    --devices=1 \
    --nodes=1 \
    --accelerator="gpu" \
    --strategy="auto" \
    --warmup_steps="0" \
    --warmup_max_steps="100000" \
    --save_interval="1000" \
    --problem_type="$PROBLEM_TYPE" \
    --cross_fold="$CROSS_FOLD" \
    --cross_fold_index="$CROSS_FOLD_INDEX" \
    --num_labels="$NUM_LABELS" \
    --task_name="$TASK_NAME" \
    --amino="True"
}

# Check if TASK is provided
if [ -z "$TASK" ]; then
  for TASK in "TransAbundance"; do
    bash shell_scripts_example/finetune.sh "$TASK"
  done
  exit
fi

# Define parameters based on the task
case "$TASK" in
  "Localization")
    PROBLEM_TYPE="single_label_classification"
    NUM_LABELS="5"
    NUM_LABELS="5"
    CROSS_FOLD="5"
    FILE_PATH="$(pwd)/data/CaLM/$TASK/dev.jsonl"
    TRAIN_FILE="$FILE_PATH"
    VALID_FILE="$FILE_PATH"
    TEST_FILE="$FILE_PATH"
    ;;
    
  "Meltome" | "ProAbundance" | "Solubility" | "TransAbundance")
    PROBLEM_TYPE="regression"
    NUM_LABELS="1"
    CROSS_FOLD="5"
    
    if [ "$TASK" == "TransAbundance" ]; then
      for SPECIES in "hvolcanii" "ecoli"; do
        SPECIFIC_TASK="${TASK}_${SPECIES}"
        FILE_PATH="$(pwd)/data/CaLM/$SPECIFIC_TASK/dev.jsonl"
        TRAIN_FILE="$FILE_PATH"
        VALID_FILE="$FILE_PATH"
        TEST_FILE="$FILE_PATH"
        for CROSS_FOLD_INDEX in $(seq 0 $((CROSS_FOLD - 1))); do
          finetune_protein_model "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE" "$PROBLEM_TYPE" "$NUM_LABELS" "$SPECIFIC_TASK" "$CROSS_FOLD" "$CROSS_FOLD_INDEX"
        done
      done
    else
      FILE_PATH="$(pwd)/data/CaLM/$TASK/dev.jsonl"
      TRAIN_FILE="$FILE_PATH"
      VALID_FILE="$FILE_PATH"
      TEST_FILE="$FILE_PATH"
      for CROSS_FOLD_INDEX in $(seq 0 $((CROSS_FOLD - 1))); do
        finetune_protein_model "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE" "$PROBLEM_TYPE" "$NUM_LABELS" "$TASK" "$CROSS_FOLD" "$CROSS_FOLD_INDEX"
      done
    fi
    ;;
    
  "Function")
    PROBLEM_TYPE="multi_label_classification"
    for file in $(pwd)/data/CaLM/$TASK/*.jsonl; do
      TRAIN_FILE="$file"
      VALID_FILE="$file"
      TEST_FILE="$file"
      base_name=$(basename "$file" .jsonl)
      split_array=(${base_name//_/ })
      NUM_LABELS=${split_array[-1]}
      finetune_protein_model "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE" "$PROBLEM_TYPE" "$NUM_LABELS" "$TASK"
    done
    ;;
    
  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac