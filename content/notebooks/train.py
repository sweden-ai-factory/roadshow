
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

# Dataset creation and preprocessing 

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl"
    }
)

def to_conversation(example):

    instruction = (example.get("instruction") or "").strip()
    context = (example.get("input") or "").strip()
    answer = (example.get("output") or "").strip()

    if instruction and context:
        user_content = (
            f"{instruction}\n\n"
            f"Context:\n{context}"
        )

    elif instruction:
        user_content = instruction

    elif context:
        user_content = context

    else:
        user_content = ""

    return {
        "messages": [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
    }


dataset = dataset.map(
    to_conversation,
    remove_columns=dataset["train"].column_names,
)
print(dataset["train"])
print(dataset["train"][0])

train_dataset = dataset["train"]
print(train_dataset[0])
eval_dataset = dataset["validation"]

# Model instantiation 

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)

training_args = SFTConfig(
    output_dir="./outputs",
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    packing=True,
    eval_strategy="epoch",
    max_length=512
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config
)

trainer.model.print_trainable_parameters()

trainer.train()

trainer.save_model("./outputs/final")
