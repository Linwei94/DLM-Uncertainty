CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset mmlu --prompt mmlu_0 --conf token_probs
CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset mmlu --prompt mmlu_0 --conf p_true
CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset mmlu --prompt mmlu_vnc --conf vnc
CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset mmlu --prompt mmlu_0 --conf semantic_uncertainty