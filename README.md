# LLM Fine-Tuning — Learning Project

A hands-on progression through the fundamentals of LLM training and fine-tuning using a tiny, fast model.

**Model**: `HuggingFaceTB/SmolLM2-135M` — 135M params, ~270 MB  
**Task**: Instruction fine-tuning (base → chat-capable model)  
**Dataset**: `HuggingFaceTB/smoltalk` (2k-sample subset)

---

## Learning Path

| Notebook | Topic | Where to run |
|---|---|---|
| [01_inference](notebooks/01_inference.ipynb) | Tokens, attention, generation params | Local CPU |
| [02_explore_dataset](notebooks/02_explore_dataset.ipynb) | Dataset format, chat templates, tokenizing | Local CPU |
| [03_finetune_lora](notebooks/03_finetune_lora.ipynb) | LoRA + SFTTrainer fine-tuning | **Google Colab (T4)** |
| [04_compare](notebooks/04_compare.ipynb) | Base vs fine-tuned vs reference Instruct | Local CPU |
| [05_push_to_hub](notebooks/05_push_to_hub.ipynb) | Publish your adapter to HF Hub | Local / Colab |

---

## Setup

```bash
pip install -r requirements.txt
```

For the training notebook, upload it to [Google Colab](https://colab.research.google.com) and select **Runtime → Change runtime type → T4 GPU**.

---

## Key Concepts Covered

- **Tokenization**: subwords, special tokens, `input_ids` / `attention_mask`
- **Causal LM inference**: logits, sampling strategies (`temperature`, `top_p`)
- **Chat templates**: how multi-turn conversation is serialized into tokens
- **LoRA**: low-rank adaptation — fine-tune ~0.5% of params, save ~5 MB adapter
- **SFT (Supervised Fine-Tuning)**: train on (prompt, response) pairs with prompt-loss masking
- **HF ecosystem**: `transformers`, `datasets`, `peft`, `trl`, `huggingface_hub`

---

## After This Project — Natural Next Steps

- **QLoRA** (4-bit quantization) — same technique on a 7B model
- **DPO** (Direct Preference Optimization) — align the model to prefer better responses
- **RLHF** — full reward-model + PPO pipeline
- **Eval harnesses** — `lm-evaluation-harness` for standardized benchmarks
