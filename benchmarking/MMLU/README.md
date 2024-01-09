# Evaluation of MMLU

This code is mainly built upon [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub), [LiteLLM](https://github.com/BerriAI/litellm) and [Gemini Benchmark](https://github.com/neulab/gemini-benchmark).

Below is the default command with `gpt-3.5-turbo` with 5-shot examples:
```bash
python run_mmlu.py
```

If you want to incorporate chain-of-thought prompting:
```bash
python run_mmlu.py --cot
```


If you are running models on your local machine, you should start a server in the backend:
```bash
# Start the server in the backend
python start_server.py \
    --model_name llama-2-7b \
    --tensor_parallel_size 4\
    --port 8964 \
    --cuda_devices 0,1,2,3

# Run the evaluation
python run_mmlu.py --model_name llama-2-7b
```
