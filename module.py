import os
import io
import json
import sys
import logging
import inspect
import subprocess
import contextlib
from datasets import load_dataset, Dataset
from IPython.display import display
from huggingface_hub import login, HfApi
from pathlib import Path
from jinja2 import Template

MODEL_NAME = None
AUTHOR = None
HF_TOKEN = None
_CURRENT_TOOL = None  # Track which tool is currently active: "unsloth" or "llamafactory"
BASE_MODEL = "Qwen/Qwen3-8B-Instruct"  # Default model, can be changed with set_model()
MODEL_TEMPLATE = "qwen"  # Default template for the model


def set(name, author):
    global MODEL_NAME, AUTHOR
    MODEL_NAME = name
    AUTHOR = author


def set_model(model_name, template=None):
    """
    Set the base model to use for fine-tuning
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B-Instruct", "ura-hcmut/GemSUra-2B")
        template: Template name for the model (e.g., "qwen", "gemma"). If None, will auto-detect based on model name
    """
    global BASE_MODEL, MODEL_TEMPLATE
    BASE_MODEL = model_name
    
    # Auto-detect template if not provided
    if template is None:
        model_lower = model_name.lower()
        if "qwen" in model_lower:
            MODEL_TEMPLATE = "qwen"
        elif "gemma" in model_lower or "gemsura" in model_lower:
            MODEL_TEMPLATE = "gemma"
        elif "llama" in model_lower:
            MODEL_TEMPLATE = "llama3"
        elif "mistral" in model_lower:
            MODEL_TEMPLATE = "mistral"
        else:
            MODEL_TEMPLATE = "qwen"  # Default to qwen
    else:
        MODEL_TEMPLATE = template


def hf(token):
    global HF_TOKEN
    HF_TOKEN = token
    login(HF_TOKEN, add_to_git_credential=True)


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
    original_dir = os.getcwd()
    try:
        os.chdir("/content/LLaMA-Factory")
        subprocess.run(["pip", "install", "-e", ".[torch,bitsandbytes]"], check=False)
    finally:
        os.chdir(original_dir)
    
    print("✅ Đã setup packages cho LLaMA-Factory xong!")
    print("=" * 80)
    print()
    
    _CURRENT_TOOL = "llamafactory"


def identify_dataset(record):
    global MODEL_NAME, AUTHOR
    record["output"] = (
        record["output"]
        .replace("Gemma-tvts", MODEL_NAME)
        .replace("Long Nguyen", AUTHOR)
    )
    return record


def preprocess_dataset(dataset, num_to_train=None):
    setup_llamafactory()
    
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
    setup_llamafactory()
    
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


def train(datasets, num_train_epochs, continue_training=True):
    # Detect if datasets is a single dataset (Unsloth style) or a list (LLaMA-Factory style)
    # If it's a list of datasets, use LLaMA-Factory; otherwise use Unsloth
    if isinstance(datasets, list):
        # LLaMA-Factory style: list of datasets
        setup_llamafactory()
        _train_llamafactory(datasets, num_train_epochs, continue_training)
    else:
        # Unsloth style: single dataset (could be combined dataset)
        setup_unsloth()
        _train_unsloth(datasets, num_train_epochs, continue_training)


def _train_llamafactory(datasets, num_train_epochs, continue_training):
    """Internal function for LLaMA-Factory training"""
    caller_locals = inspect.stack()[2][0].f_locals  # Skip train() and _train_llamafactory()
    dataset_names = ",".join(
        [
            name
            for dataset in datasets
            for name, val in caller_locals.items()
            if val is dataset
        ]
    )

    if not continue_training:
        os.system("rm -rf /content/LLaMA-Factory/lora_model")

    global BASE_MODEL, MODEL_TEMPLATE
    
    args = dict(
        stage="sft",
        do_train=True,
        model_name_or_path=BASE_MODEL,
        dataset=dataset_names,
        template=MODEL_TEMPLATE,
        finetuning_type="lora",
        lora_target="all",
        output_dir="lora_model",
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

    file_path = "/content/LLaMA-Factory/train_config.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(args, f, ensure_ascii=False, indent=4)

    original_dir = os.getcwd()
    try:
        os.chdir("/content/LLaMA-Factory")
        process = subprocess.Popen(
            ["llamafactory-cli", "train", "train_config.json"],
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
    finally:
        os.chdir(original_dir)


def _train_unsloth(dataset, num_train_epochs, continue_training):
    """Internal function for Unsloth training"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    global BASE_MODEL
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    
    if not continue_training:
        # Reset adapter if needed
        pass
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field="output",
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir="outputs",
            optim="adamw_8bit",
        ),
    )
    
    trainer.train()
    
    # Save the adapter
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")


class SuppressLogging:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.disable(logging.NOTSET)


def test():
    setup_llamafactory()
    
    original_dir = os.getcwd()
    try:
        os.chdir("/content/LLaMA-Factory/src")
        from llamafactory.chat import ChatModel
        from llamafactory.extras.misc import torch_gc
        os.chdir("/content/LLaMA-Factory")

        global BASE_MODEL, MODEL_TEMPLATE
        
        args = dict(
            model_name_or_path=BASE_MODEL,
            adapter_name_or_path="lora_model",
            template=MODEL_TEMPLATE,
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
    finally:
        os.chdir(original_dir)


def merge_and_push(repo_id):
    setup_llamafactory()
    
    original_dir = os.getcwd()
    try:
        os.chdir("/content/LLaMA-Factory/")

        global BASE_MODEL, MODEL_TEMPLATE
        
        args = dict(
            model_name_or_path=BASE_MODEL,
            adapter_name_or_path="lora_model",
            template=MODEL_TEMPLATE,
            finetuning_type="lora",
            export_dir="lora_model_merged",
            export_size=2,
            export_device="cpu",
        )

        with open("lora_model_merged.json", "w", encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=False, indent=2)

        with SuppressLogging(), open(os.devnull, "w") as devnull:
            subprocess.run(
                ["llamafactory-cli", "export", "lora_model_merged.json"],
                stdout=devnull,
                stderr=devnull,
                check=True,
            )

        print("***** Đã merge model thành công và tiến hành upload lên Huggingface! *****")

        model_dir = "/content/LLaMA-Factory/lora_model_merged"
        tokenizer_dir = "/content/LLaMA-Factory/lora_model"

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
    finally:
        os.chdir(original_dir)


model = None
tokenizer = None
messages = []


def inference(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
    setup_unsloth()
    
    logging.getLogger().setLevel(logging.ERROR)
    global model, tokenizer

    try:
        from unsloth import FastLanguageModel
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
    setup_unsloth()
    
    global model, tokenizer, messages, MODEL_TEMPLATE

    # Determine if we should use tokenizer's chat template
    use_tokenizer_template = False
    if hasattr(tokenizer, 'apply_chat_template') and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            # Test if tokenizer template works
            test_messages = [{"role": "user", "content": "test"}]
            tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            use_tokenizer_template = True
        except:
            use_tokenizer_template = False
    
    # Fallback custom template based on model type
    if not use_tokenizer_template:
        if MODEL_TEMPLATE == "qwen" or MODEL_TEMPLATE == "qwen3":
            chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n<|im_start|>assistant\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"""
        elif MODEL_TEMPLATE == "gemma":
            chat_template = """{{ '<bos>' }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"""
        else:
            # Default to Qwen template
            chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n<|im_start|>assistant\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"""

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

        if use_tokenizer_template:
            # Use tokenizer's built-in chat template
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Use custom template (chat_template is defined in the if block above)
            template = Template(chat_template)
            input_text = template.render(messages=messages)

        print(f"Trợ lý: ", end="", flush=True)

        inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True
        )

        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response based on model type
        if MODEL_TEMPLATE == "qwen" or MODEL_TEMPLATE == "qwen3":
            if "<|im_start|>assistant" in decoded_text:
                response = decoded_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            else:
                response = decoded_text.strip()
        elif MODEL_TEMPLATE == "gemma":
            if "model" in decoded_text:
                response = decoded_text.split("model")[-1].strip()
            else:
                response = decoded_text.strip()
        else:
            # Default extraction
            response = decoded_text.strip()
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        
        print(response)

        if history:
            messages.append({"role": "assistant", "content": response})


def quantize_and_push(repo_id):
    setup_unsloth()
    
    logging.getLogger("unsloth").setLevel(logging.CRITICAL)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    temp_stdout = io.StringIO()
    original_dir = os.getcwd()
    
    try:
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
    finally:
        os.chdir(original_dir)


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