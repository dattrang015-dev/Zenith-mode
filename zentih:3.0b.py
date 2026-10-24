# =====================================================================
# BƯỚC 1: CÀI ĐẶT VÀ ĐỒNG BỘ THƯ VIỆN PHẦN CỨNG
# =====================================================================
print("🛠️ Đang cấu hình lại các thư viện để đồng bộ phần cứng...")
!pip install "setuptools<82" -q
!pip install transformers trl peft bitsandbytes datasets accelerate -q

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig
from trl import SFTTrainer

# =====================================================================
# BƯỚC 2: CẤU HÌNH BẢO MẬT VÀ TÀI KHOẢN HUGGING FACE
# =====================================================================
HF_TOKEN = "hf key của bạn"
os.environ["HF_TOKEN"] = HF_TOKEN

# =====================================================================
# BƯỚC 3: CHỌN MÔ HÌNH NỀN MẠNH NHẤT ĐỂ ĐẠI DIỆN CHO CẢ HAI
# =====================================================================
list_models = [
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct"
]

# Sử dụng mô hình Qwen2.5-3B làm cốt lõi đại diện tốt nhất
chon_index = 1
model_id = list_models[chon_index]

print(f"\n🚀 Đang khởi tạo nền tảng huấn luyện tích hợp mô hình: {model_id}")

# Tải Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Cấu hình nén 4-bit QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Cấu hình ma trận tinh chỉnh LoRA
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# =====================================================================
# BƯỚC 4: TẬP DỮ LIỆU ĐA NGÔN NGỮ VẠN NĂNG HỢP NHẤT
# =====================================================================
chat_data = [
    # Tiếng Việt
    [{"role": "user", "content": "Chào bạn, bạn là ai?"}, {"role": "assistant", "content": "Xin chào! Tôi là Zenith VN, trợ lý trí tuệ nhân tạo thông minh của bạn." }],
    [{"role": "user", "content": "Ai đã tạo ra Zenith VN?"}, {"role": "assistant", "content": "Zenith VN được phát triển và tạo ra bởi udVietNamspeedrace, chủ nhân của kênh YouTube udVietNamspeedrace." }],
    # Tiếng Anh
    [{"role": "user", "content": "Hello, who are you?"}, {"role": "assistant", "content": "Hello! I am Zenith VN, your intelligent AI assistant." }],
    [{"role": "user", "content": "Who created you?"}, {"role": "assistant", "content": "I was created by udVietNamspeedrace, the owner of the YouTube channel udVietNamspeedrace." }]
]

# Áp dụng template hội thoại
formatted_texts = []
for messages in chat_data:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    formatted_texts.append(text)

dataset = Dataset.from_dict({"text": formatted_texts})

def preprocess_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"], keep_in_memory=True)

print(f"🧠 Đang tải mô hình nền {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.float16
)

for name, param in model.named_parameters():
    if "norm" in name or "embed" in name:
        param.data = param.data.to(torch.float16)

# =====================================================================
# BƯỚC 6: CẤU HÌNH VÀ THỰC THI HUẦN LUYỆN (SFT TRAINER)
# =====================================================================
output_dir = "./results_zenith_local"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=30,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=1,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    report_to="none",
    push_to_hub=False,
    remove_unused_columns=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    args=training_args,
)

trainer.train()

# =====================================================================
# BƯỚC 7: ĐÓNG GÓI VÀ LƯU MÔ HÌNH
# =====================================================================
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n🎉 [HOÀN TẤT] Trọng số bộ não Zenith ({model_id}) đã được lưu tại {output_dir}!")
