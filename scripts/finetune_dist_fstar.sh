rank=$1;
mkdir -p logs;
accelerate launch --machine_rank $rank \
        src/finetune.py \
        --languages_to_train_on fstar 2>&1 | tee logs/finetune-fstar${rank}.log