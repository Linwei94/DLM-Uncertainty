CUDA_VISIBLE_DEVICES=0 python main.py --model Meta-Llama-3-8B-Instruct --dataset truthful_qa --prompt qa --conf token_probs
CUDA_VISIBLE_DEVICES=0 python main.py --model Meta-Llama-3-8B-Instruct --dataset truthful_qa --prompt qa --conf p_true
CUDA_VISIBLE_DEVICES=0 python main.py --model Meta-Llama-3-8B-Instruct --dataset truthful_qa --prompt vnc --conf vnc
CUDA_VISIBLE_DEVICES=0 python main.py --model Meta-Llama-3-8B-Instruct --dataset truthful_qa --prompt qa --conf semantic_uncertainty

CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset truthful_qa --prompt qa --conf token_probs
CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset truthful_qa --prompt qa --conf p_true
CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset truthful_qa --prompt vnc --conf vnc
CUDA_VISIBLE_DEVICES=0 python main.py --model LLaDA-8B-Instruct --dataset truthful_qa --prompt qa --conf semantic_uncertainty