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


# ============================================
# CÁC HÀM HỖ TRỢ UNSLOTH
# ============================================

def train(dataset, num_train_epochs=3, max_seq_length=2048, continue_training=True):
    """
    Train model với unsloth
    
    Args:
        dataset: Dataset object từ datasets (cần có columns "input" và "output")
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


# ============================================
# CÁC HÀM HỖ TRỢ LLaMA-FACTORY
# ============================================

def preprocess_dataset(dataset, num_to_train=None):
    """Preprocess dataset cho LLaMA-Factory"""
    dataset_df = dataset.to_pandas()
    if num_to_train is not None:
        dataset_df = dataset_df.head(num_to_train)
    dataset_df["input"] = dataset_df["input"].fillna("")
    caller_locals = inspect.stack()[1][0].f_locals
    dataset_name = [name for name, val in caller_locals.items() if val is dataset][0]
    file_path = f"/content/LLaMA-Factory/data/{dataset_name}.json"
    dataset_df.to_json(file_path, orient="records", force_ascii=False, indent=4)
    return file_path


def dataset_info(*datasets):
    """Tạo file dataset_info.json cho LLaMA-Factory"""
    info = {}
    for dataset in datasets:
        caller_locals = inspect.stack()[1][0].f_locals
        dataset_name = [name for name, val in caller_locals.items() if val is dataset][
            0
        ]
        info[dataset_name] = {"file_name": f"{dataset_name}.json"}
    file_path = "/content/LLaMA-Factory/data/dataset_info.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return file_path


def train_llamafactory(datasets, num_train_epochs, continue_training=True):
    """Train model với LLaMA-Factory"""
    caller_locals = inspect.stack()[1][0].f_locals
    dataset_names = ",".join(
        [
            name
            for dataset in datasets
            for name, val in caller_locals.items()
            if val is dataset
        ]
    )

    if not continue_training:
        os.system("rm -rf /content/LLaMA-Factory/qwen_lora")

    args = dict(
        stage="sft",
        do_train=True,
        model_name_or_path="Qwen/Qwen3-8B",
        dataset=dataset_names,
        template="qwen3",
        finetuning_type="lora",
        lora_target="all",
        output_dir="qwen_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        warmup_ratio=0.1,
        save_steps=1000,
        learning_rate=5e-5,
        num_train_epochs=num_train_epochs,
        max_samples=500,
        max_grad_norm=1.0,
        quantization_bit=4,
        loraplus_lr_ratio=16.0,
        fp16=True,
    )

    file_path = "/content/LLaMA-Factory/train_qwen.json"
    
    # Xóa file config cũ nếu tồn tại để tránh dùng model name cũ
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(args, f, ensure_ascii=False, indent=4)

    print("=" * 80)
    print("🚀 BẮT ĐẦU TRAINING VỚI LLaMA-FACTORY")
    print("=" * 80)
    print(f"📊 Model: {args['model_name_or_path']}")
    print(f"📁 Dataset: {dataset_names}")
    print(f"🔄 Epochs: {num_train_epochs}")
    print(f"📦 Batch size: {args['per_device_train_batch_size']} x {args['gradient_accumulation_steps']}")
    print(f"📈 Learning rate: {args['learning_rate']}")
    print(f"💾 Output dir: {args['output_dir']}")
    print("=" * 80)
    print()

    os.chdir("/content/LLaMA-Factory")

    subprocess.run(["pip", "install", "-e", ".[torch,bitsandbytes]"], check=True)
    process = subprocess.Popen(
        ["llamafactory-cli", "train", "train_qwen.json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    start_printing_all = False

    for line in iter(process.stdout.readline, b""):
        decoded_line = line.decode()
        if "train metrics" in decoded_line.lower():
            start_printing_all = True
        if "loss" in decoded_line.lower() or start_printing_all:
            print(decoded_line, end="")

    process.stdout.close()
    return_code = process.wait()
    
    print()
    print("=" * 80)
    if return_code == 0:
        print("✅ TRAINING HOÀN THÀNH!")
    else:
        print("❌ TRAINING CÓ LỖI! (Return code:", return_code, ")")
    print("=" * 80)


def test_llamafactory():
    """Test model đã được train với LLaMA-Factory"""
    os.chdir("/content/LLaMA-Factory/src")
    from llamafactory.chat import ChatModel
    from llamafactory.extras.misc import torch_gc

    os.chdir("/content/LLaMA-Factory")

    args = dict(
        model_name_or_path="Qwen/Qwen3-8B",
        adapter_name_or_path="qwen_lora",
        template="qwen3",
        finetuning_type="lora",
        quantization_bit=4,
    )

    with SuppressLogging():
        chat_model = ChatModel(args)

    print("***** Nhập clear để xóa lịch sử trò chuyện, nhập exit để thoát nha! *****")
    messages = []
    while True:
        query = input("\nNgười dùng: ")
        if query.strip().lower() == "exit":
            break
        if query.strip().lower() == "clear":
            messages = []
            torch_gc()
            print("Lịch sử trò chuyện vừa được xóa.")
            continue

        messages.append({"role": "user", "content": query})
        print(f"Trợ lý: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})
    torch_gc()


def merge_and_push_llamafactory(repo_id):
    """Merge và push model lên Hugging Face với LLaMA-Factory"""
    os.chdir("/content/LLaMA-Factory/")

    args = dict(
        model_name_or_path="Qwen/Qwen3-8B",
        adapter_name_or_path="qwen_lora",
        template="qwen3",
        finetuning_type="lora",
        export_dir="qwen_lora_merged",
        export_size=2,
        export_device="cpu",
    )

    with open("qwen_lora_merged.json", "w", encoding="utf-8") as f:
        json.dump(args, f, ensure_ascii=False, indent=2)

    with SuppressLogging(), open(os.devnull, "w") as devnull:
        subprocess.run(
            ["llamafactory-cli", "export", "qwen_lora_merged.json"],
            stdout=devnull,
            stderr=devnull,
            check=True,
        )

    print("***** Đã merge model thành công và tiến hành upload lên Huggingface! *****")

    model_dir = "/content/LLaMA-Factory/qwen_lora_merged"
    tokenizer_dir = "/content/LLaMA-Factory/qwen_lora"

    tokenizer_config_path = Path(tokenizer_dir) / "tokenizer_config.json"
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    tokenizer_config.pop("chat_template", None)
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    api = HfApi()
    global HF_TOKEN

    for file in os.listdir(model_dir):
        file_path = Path(model_dir) / file
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )

    for file_name in tokenizer_files:
        file_path = Path(tokenizer_dir) / file_name
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )


class SuppressLogging:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)


def test():
    """Test model đã được train với unsloth"""
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
    """Merge LoRA weights vào base model và push lên Hugging Face với unsloth"""
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
    except Exception as e:
        print("Bạn chỉ cần chạy inference một lần duy nhất, bạn không cần chạy lại!")


def chat(max_new_tokens=128, history=True):
    global model, tokenizer, messages

    chat_template = """{{ '<bos>' }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"""

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

        inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True
        )

        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "model" in decoded_text:
            response = decoded_text.split("model")[-1].strip()
        else:
            response = decoded_text.strip()
        print(response)

        if history:
            messages.append({"role": "assistant", "content": response})


def quantize_and_push(repo_id):
    logging.getLogger("unsloth").setLevel(logging.CRITICAL)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    temp_stdout = io.StringIO()
    os.chdir("/content")

    global model, tokenizer, HF_TOKEN
    try:
        with contextlib.redirect_stdout(temp_stdout), contextlib.redirect_stderr(
            temp_stdout
        ):
            model.push_to_hub_gguf(
                repo_id, tokenizer, token=HF_TOKEN
            )
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        return
    finally:
        temp_stdout.seek(0)
        output_lines = temp_stdout.readlines()

    sys.stdout = original_stdout
    sys.stderr = original_stderr

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