# Self Alignment Implementation

A straightforward implementation of paper "Self-Alignment with Instruction Backtranslation." (DSAA 6000Q)

## Dataset and Model Weight Link

**Full dataset of PairRM_Preference_LIMA can be accessed at:** [GPRM/PairRM_Preference_LIMA](https://huggingface.co/datasets/GPRM/PairRM_Preference_LIMA)

**Lora DPO model weight can be accessed at:** [GPRM/Mistral-7B-PairRM-DPO](https://huggingface.co/GPRM/Mistral-7B-PairRM-DPO)

## Key File Descriptions

The training code for supervise-finetuning LLaMA2-7B is based on LLaMA Factory Repo.

### `6000Q/`

All python scripts and output examples are stored here.

- `generate_instructions.ipynb`: Combine LoRA weights with the base model (to obtain full Llama-2-7b-OpenAssistant-Backwards), and generate instructions based on Llama-2-7b-OpenAssistant-Backwards with LIMA dataset (I use the full training set here). Print out 5 examples of generated instructions.

- `self_curation.ipynb`: Use In-context Learning with rating criteria to judge and filter positive/negative samples (LLaMA2-7B-chat), and formulate high-quality data. Output 5 positive samples and 5 negative samples.

- `final_LIMA_filtered_SFT.ipynb`: Supervise-finetuning LLaMA2-7B on filtered LIMA QA pairs, and output 5 example results.

- `push_to_hf.ipynb`: Push trained models and dataset to HuggingFace.

### `data/`

All data and generated files during inference are stored here.

- `lima/LIMA_Scored_Samples/lima_train_full_scored.jsonl`: Final filtered high-quality data of LIMA.

### `examples/`

- `train_lora/llama2_lora_sft_ds3_openassist.yaml`: yaml file to train Llama-2-7b-OpenAssistant-Backwards model.

- `train_lora/llama2_lora_sft_ds3_final_alignment.yaml`: yaml file to train Llama-2-7b-LIMA-Alignment model.