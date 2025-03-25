# PoPAI-RL

1. Download and extract the dataset from https://microsoft-my.sharepoint.com/:u:/p/saikatc/EXMx8d_47ktCpSTvRyweQn0BnQCbBfBpRzvQzEVzyjDXzA?e=PLm8I4

2. Install the dependencies from `requirements.txt`
3. Run finetuning script from 
```
python finetune.py \
    --data_path <path where you extracted the data>/qwen_prompt_data \
    --four_bit \
    --use_peft  \
    --output_dir <output dir to save checkpoints>
```