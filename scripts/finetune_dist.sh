rank=$1;
mkdir -p logs;
accelerate launch \
        --machine_rank $rank \
        src/finetune.py 2>&1 | tee logs/finetune-${rank}.log