import os
import io
import json
import torch
import sys
import logging
import inspect
import subprocess
import contextlib
# Import động để tránh lỗi khi chưa cài đặt
from datasets import load_dataset, Dataset
from IPython.display import display
from huggingface_hub import login, HfApi
from pathlib import Path
from jinja2 import Template

# FastLanguageModel sẽ được import động trong các hàm cần dùng

MODEL_NAME = None
AUTHOR = None
HF_TOKEN = None
_CURRENT_TOOL = None  # Track tool đang được dùng: "unsloth" hoặc "llamafactory"


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
# HÀM QUẢN LÝ PACKAGES ĐỘNG
# ============================================

def setup_unsloth(force_reinstall=False):
    """
    Setup packages cho Unsloth - uninstall LLaMA-Factory và install packages cho Unsloth
    
    Args:
        force_reinstall: Nếu True, sẽ reinstall lại ngay cả khi đã setup
    """
    global _CURRENT_TOOL
    
    if _CURRENT_TOOL == "unsloth" and not force_reinstall:
        print("✅ Unsloth đã được setup sẵn, bỏ qua...")
        return
    
    print("=" * 80)
    print("🔧 ĐANG SETUP PACKAGES CHO UNSLOTH")
    print("=" * 80)
    
    # Uninstall LLaMA-Factory nếu có
    print("\n📦 Đang uninstall LLaMA-Factory...")
    try:
        subprocess.run(["pip", "uninstall", "-y", "llamafactory-cli", "llamafactory"], 
                      check=False, capture_output=True)
        print("✅ Đã uninstall LLaMA-Factory")
    except:
        pass
    
    # Install packages cho Unsloth
    print("\n📦 Đang install packages cho Unsloth...")
    unsloth_packages = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "trl",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
    ]
    
    for package in unsloth_packages:
        try:
            subprocess.run(["pip", "install", "-q", "--upgrade", package], check=False)
        except:
            pass
    
    print("✅ Đã setup packages cho Unsloth xong!")
    print("=" * 80)
    print()
    
    _CURRENT_TOOL = "unsloth"


def setup_llamafactory(force_reinstall=False):
    """
    Setup packages cho LLaMA-Factory - uninstall Unsloth và install packages cho LLaMA-Factory
    
    Args:
        force_reinstall: Nếu True, sẽ reinstall lại ngay cả khi đã setup
    """
    global _CURRENT_TOOL
    
    if _CURRENT_TOOL == "llamafactory" and not force_reinstall:
        print("✅ LLaMA-Factory đã được setup sẵn, bỏ qua...")
        return
    
    print("=" * 80)
    print("🔧 ĐANG SETUP PACKAGES CHO LLaMA-FACTORY")
    print("=" * 80)
    
    # Uninstall Unsloth nếu có
    print("\n📦 Đang uninstall Unsloth...")
    try:
        subprocess.run(["pip", "uninstall", "-y", "unsloth"], 
                      check=False, capture_output=True)
        print("✅ Đã uninstall Unsloth")
    except:
        pass
    
    # Uninstall trl nếu có (có thể conflict với LLaMA-Factory)
    try:
        subprocess.run(["pip", "uninstall", "-y", "trl"], 
                      check=False, capture_output=True)
    except:
        pass
    
    # Clone LLaMA-Factory nếu chưa có
    if not os.path.exists("/content/LLaMA-Factory"):
        print("\n📥 Đang clone LLaMA-Factory...")
        subprocess.run(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", "/content/LLaMA-Factory"], 
                      check=False)
        print("✅ Đã clone LLaMA-Factory")
    
    # Install LLaMA-Factory
    print("\n📦 Đang install LLaMA-Factory...")
    os.chdir("/content/LLaMA-Factory")
    subprocess.run(["pip", "install", "-e", ".[torch,bitsandbytes]"], check=False)
    os.chdir("/content")
    
    print("✅ Đã setup packages cho LLaMA-Factory xong!")
    print("=" * 80)
    print()
    
    _CURRENT_TOOL = "llamafactory"


def preprocess_dataset(dataset, num_to_train=None):
    """Preprocess dataset cho LLaMA-Factory - tự động setup packages"""
    setup_llamafactory()
    
    dataset_df = dataset.to_pandas()
    if num_to_train is not None:
        dataset_df = dataset_df.head(num_to_train)
    dataset_df["input"] = dataset_df["input"].fillna("")
    caller_locals = inspect.stack()[1][0].f_locals
    dataset_name = [name for name, val in caller_locals.items() if val is dataset][0]
    
    # Tạo thư mục data nếu chưa có
    os.makedirs("/content/LLaMA-Factory/data", exist_ok=True)
    
    file_path = f"/content/LLaMA-Factory/data/{dataset_name}.json"
    dataset_df.to_json(file_path, orient="records", force_ascii=False, indent=4)
    return file_path


def dataset_info(*datasets):
    """Tạo file dataset_info.json cho LLaMA-Factory - tự động setup packages"""
    setup_llamafactory()
    
    info = {}
    for dataset in datasets:
        caller_locals = inspect.stack()[1][0].f_locals
        dataset_name = [name for name, val in caller_locals.items() if val is dataset][
            0
        ]
        info[dataset_name] = {"file_name": f"{dataset_name}.json"}
    
    # Tạo thư mục data nếu chưa có
    os.makedirs("/content/LLaMA-Factory/data", exist_ok=True)
    
    file_path = "/content/LLaMA-Factory/data/dataset_info.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return file_path


def train(datasets, num_train_epochs, continue_training=True):
    """Train model với LLaMA-Factory - tự động setup packages"""
    setup_llamafactory()
    
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
        os.system("rm -rf /content/LLaMA-Factory/gemma_lora")

    args = dict(
        stage="sft",
        do_train=True,
        model_name_or_path="ura-hcmut/GemSUra-2B",
        dataset=dataset_names,
        template="gemma",
        finetuning_type="lora",
        lora_target="all",
        output_dir="gemma_lora",
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

    file_path = "/content/LLaMA-Factory/train_gemma.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(args, f, ensure_ascii=False, indent=4)

    os.chdir("/content/LLaMA-Factory")
    
    # Đảm bảo LLaMA-Factory đã được cài đặt (đã setup trong setup_llamafactory())
    # Chỉ cài lại nếu cần thiết
    process = subprocess.Popen(
        ["llamafactory-cli", "train", "train_gemma.json"],
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
    process.wait()


class SuppressLogging:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)


def test():
    """Test model đã được train với LLaMA-Factory - tự động setup packages"""
    setup_llamafactory()
    
    os.chdir("/content/LLaMA-Factory/src")
    from llamafactory.chat import ChatModel
    from llamafactory.extras.misc import torch_gc

    os.chdir("/content/LLaMA-Factory")

    args = dict(
        model_name_or_path="ura-hcmut/GemSUra-2B",
        adapter_name_or_path="gemma_lora",
        template="gemma",
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


def merge_and_push(repo_id):
    """Merge và push model lên Hugging Face với LLaMA-Factory - tự động setup packages"""
    setup_llamafactory()
    
    os.chdir("/content/LLaMA-Factory/")

    args = dict(
        model_name_or_path="ura-hcmut/GemSUra-2B",
        adapter_name_or_path="gemma_lora",
        template="gemma",
        finetuning_type="lora",
        export_dir="gemma_lora_merged",
        export_size=2,
        export_device="cpu",
    )

    with open("gemma_lora_merged.json", "w", encoding="utf-8") as f:
        json.dump(args, f, ensure_ascii=False, indent=2)

    with SuppressLogging(), open(os.devnull, "w") as devnull:
        subprocess.run(
            ["llamafactory-cli", "export", "gemma_lora_merged.json"],
            stdout=devnull,
            stderr=devnull,
            check=True,
        )

    print("***** Đã merge model thành công và tiến hành upload lên Huggingface! *****")

    model_dir = "/content/LLaMA-Factory/gemma_lora_merged"
    tokenizer_dir = "/content/LLaMA-Factory/gemma_lora"

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


model = None
tokenizer = None
messages = []


def inference(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
    """Load model với unsloth để inference - tự động setup packages"""
    setup_unsloth()
    
    # Import động sau khi setup
    from unsloth import FastLanguageModel
    
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
    """Chat với model đã load bằng unsloth - tự động setup packages"""
    setup_unsloth()
    
    # Import động sau khi setup
    from unsloth import FastLanguageModel
    
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
    """Quantize model sang GGUF format và push lên Hugging Face - tự động setup packages"""
    setup_unsloth()
    
    # Import động sau khi setup
    from unsloth import FastLanguageModel
    
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