CUDA_VISIBLE_DEVICES=3 python main.py --model Meta-Llama-3-8B-Instruct --dataset mmlu --prompt mmlu_0 --conf token_probs
CUDA_VISIBLE_DEVICES=3 python main.py --model Meta-Llama-3-8B-Instruct --dataset mmlu --prompt mmlu_0 --conf p_true
CUDA_VISIBLE_DEVICES=3 python main.py --model Meta-Llama-3-8B-Instruct --dataset mmlu --prompt mmlu_vnc --conf vnc
CUDA_VISIBLE_DEVICES=3 python main.py --model Meta-Llama-3-8B-Instruct --dataset mmlu --prompt mmlu_0 --conf semantic_uncertainty