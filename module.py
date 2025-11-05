import os
import io
import json
import torch
import sys
import logging
import inspect
import subprocess
import contextlib
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from IPython.display import display
from huggingface_hub import login, HfApi
from pathlib import Path
from jinja2 import Template
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = None
AUTHOR = None
HF_TOKEN = None


def set(name, author):
    global MODEL_NAME, AUTHOR
    MODEL_NAME = name
    AUTHOR = author


def hf(token):
    global HF_TOKEN
    HF_TOKEN = token
    login(HF_TOKEN, add_to_git_credential=True)


def identify_dataset(record):
    global MODEL_NAME, AUTHOR
    record["output"] = (
        record["output"]
        .replace("Gemma-tvts", MODEL_NAME)
        .replace("Long Nguyen", AUTHOR)
    )
    return record




def train(dataset, num_train_epochs=3, max_seq_length=2048, continue_training=True):
    """
    Train model với unsloth
    
    Args:
        dataset: Dataset object từ datasets
        num_train_epochs: Số epochs để train
        max_seq_length: Độ dài tối đa của sequence
        continue_training: Nếu True, tiếp tục train từ checkpoint cũ
    """
    global model, tokenizer
    
    model_name = "Qwen/Qwen3-8B"
    
    print("=" * 80)
    print("🚀 BẮT ĐẦU TRAINING VỚI UNSLOTH")
    print("=" * 80)
    print(f"📊 Model: {model_name}")
    print(f"🔄 Epochs: {num_train_epochs}")
    print(f"📏 Max sequence length: {max_seq_length}")
    print("=" * 80)
    print()
    
    # Load model với unsloth và cấu hình LoRA
    print("📦 Đang load model...")
    model, tokenizer = FastLanguageModel.get_peft_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✅ Đã load model xong!\n")
    
    # Format dataset cho unsloth
    print("📝 Đang format dataset...")
    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(inputs, outputs):
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n"
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("✅ Đã format dataset xong!\n")
    
    # Training arguments
    print("🎯 Đang khởi động training...\n")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="epoch",
            save_total_limit=3,
        ),
    )
    
    # Train
    trainer_stats = trainer.train()
    print("\n" + "=" * 80)
    print("✅ TRAINING HOÀN THÀNH!")
    print("=" * 80)
    
    # Save model
    model.save_pretrained("qwen_lora")
    tokenizer.save_pretrained("qwen_lora")
    print("\n💾 Đã lưu model vào 'qwen_lora'")
    
    return model, tokenizer


class SuppressLogging:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)


def test():
    """Test model đã được train"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print("⚠️ Model chưa được load. Đang load model từ 'qwen_lora'...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen_lora",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Đã load model xong!\n")
    
    print("***** Nhập clear để xóa lịch sử trò chuyện, nhập exit để thoát nha! *****")
    messages = []
    
    while True:
        query = input("\nNgười dùng: ")
        if query.strip().lower() == "exit":
            break
        if query.strip().lower() == "clear":
            messages = []
            print("Lịch sử trò chuyện vừa được xóa.")
            continue

        # Format prompt
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        print(f"Trợ lý: ", end="", flush=True)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            use_cache=True,
            do_sample=True,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract response after "assistant"
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        print(response)
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": response})


def merge_and_push(repo_id):
    """Merge LoRA weights vào base model và push lên Hugging Face"""
    global model, tokenizer, HF_TOKEN
    
    if model is None:
        print("⚠️ Model chưa được load. Đang load model từ 'qwen_lora'...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen_lora",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        print("✅ Đã load model xong!\n")
    
    print("🔄 Đang merge LoRA weights vào base model...")
    model.save_pretrained_merged("qwen_lora_merged", tokenizer, save_method="merged_16bit")
    print("✅ Đã merge xong!\n")
    
    print(f"📤 Đang push model lên {repo_id}...")
    model.push_to_hub(repo_id, token=HF_TOKEN, save_method="merged_16bit")
    tokenizer.push_to_hub(repo_id, token=HF_TOKEN)
    print(f"✅ Đã push model lên {repo_id} thành công!")


model = None
tokenizer = None
messages = []


def inference(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
    """Load model với unsloth để inference"""
    logging.getLogger().setLevel(logging.ERROR)
    global model, tokenizer

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        print(f"✅ Đã load model {model_name} thành công!")
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        print("Bạn chỉ cần chạy inference một lần duy nhất, bạn không cần chạy lại!")


def chat(max_new_tokens=128, history=True):
    global model, tokenizer, messages

    chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n<|im_start|>assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}"""

    messages = []

    while True:
        query = input("\nNgười dùng: ")
        if query.strip().lower() == "exit":
            break
        if query.strip().lower() == "clear":
            messages = []
            print("Lịch sử trò chuyện vừa được xóa.")
            continue

        if history:
            messages.append({"role": "user", "content": query})
        else:
            messages = [{"role": "user", "content": query}]

        template = Template(chat_template)
        input_text = template.render(messages=messages)

        print(f"Trợ lý: ", end="", flush=True)

        inputs = tokenizer(input_text, return_tensors="pt")
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True
        )

        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in decoded_text:
            response = decoded_text.split("assistant")[-1].strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
        else:
            response = decoded_text.strip()
        print(response)

        if history:
            messages.append({"role": "assistant", "content": response})


def quantize_and_push(repo_id):
    """
    Quantize model sang GGUF format và push lên Hugging Face Hub
    """
    logging.getLogger("unsloth").setLevel(logging.CRITICAL)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    temp_stdout = io.StringIO()

    global model, tokenizer, HF_TOKEN
    
    try:
        with contextlib.redirect_stdout(temp_stdout), contextlib.redirect_stderr(
            temp_stdout
        ):
            # Push model lên Hub ở dạng GGUF (quantized)
            model.push_to_hub_gguf(repo_id, tokenizer, token=HF_TOKEN)
            print(f"***** Đã quantize và push model lên {repo_id} thành công! *****")
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Lỗi khi push model lên Hub: {e}")
        return
    finally:
        temp_stdout.seek(0)
        output_lines = temp_stdout.readlines()

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Print output if needed
    start_printing = False
    for line in output_lines:
        if "main: quantize time" in line.lower():
            start_printing = True
        if start_printing:
            print(line, end="")


def thank_you_and_good_luck():
    art = [
        "⠀⠀⠀⠀⠀⠀⢀⣰⣀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⢀⣀⠀⠀⠀⢀⣄⠘⠀⠀⣶⡿⣷⣦⣾⣿⣧",
        "⢺⣾⣶⣦⣰⡟⣿⡇⠀⠀⠻⣧⠀⠛⠀⡘⠏",
        "⠈⢿⡆⠉⠛⠁⡷⠁⠀⠀⠀⠉⠳⣦⣮⠁⠀",
        "⠀⠀⠛⢷⣄⣼⠃⠀⠀⠀⠀⠀⠀⠉⠀⠠⡧",
        "⠀⠀⠀⠀⠉⠋⠀⠀⠀⠠⡥⠄⠀⠀⠀⠀⠀",
        "",
        "Chúc các bạn có một trải nghiệm tuyệt vời và đáng nhớ tại Trại hè CSE Summer School 2024 nhé!",
    ]

    for line in art:
        print(line)
