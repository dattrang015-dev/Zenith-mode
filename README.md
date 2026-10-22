# Zenith Mode

Zenith Mode là bộ script Python dùng để huấn luyện và tùy chỉnh mô hình ngôn ngữ Qwen theo phong cách hội thoại đa ngôn ngữ, tập trung vào việc tạo ra trợ lý AI mang tên Zenith VN.

Dự án này bao gồm các file huấn luyện cho hai phiên bản mô hình:

- `zenith:0.5b.py` — phiên bản Qwen 1.5 0.5B
- `zenith:1.5b.py` — phiên bản Qwen 1.5 1.8B

## Mục tiêu

- Tạo ra một mô hình chatbot có khả năng trả lời theo phong cách tự nhiên
- Hỗ trợ nhiều ngôn ngữ và nền văn hóa khác nhau
- Dùng LoRA để fine-tune hiệu quả trên các GPU có tài nguyên hạn chế
- Tương thích với Hugging Face Transformers, PEFT, TRL và BitsAndBytes

## Cấu trúc dự án

```text
Zenith-mode/
├── README.md
├── zenith:0.5b.py
├── zenith:1.5b.py
└── results_zenith_local/    # do người dùng tạo ra sau khi chạy training
```

## Công nghệ chính

- Python 3
- Hugging Face Transformers
- PEFT
- TRL
- BitsAndBytes
- PyTorch
- Datasets
- Accelerate

## Tính năng của script

- Tải mô hình `Qwen/Qwen1.5-0.5B-Chat` hoặc `Qwen/Qwen1.5-1.8B-Chat`
- Dùng `BitsAndBytesConfig` để load 4-bit quantization
- Cấu hình `LoRA` với `target_modules` như `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Tạo dataset hội thoại dựa trên `chat template`
- Huấn luyện bằng `SFTTrainer`
- Lưu model ra thư mục output tương ứng

## Yêu cầu môi trường

Trước khi chạy, bạn cần:

1. Một máy tính hoặc notebook có GPU hỗ trợ
2. Python 3.10+
3. Access token Hugging Face hợp lệ nếu model cần đăng nhập
4. Cài đặt các gói sau:

```bash
pip install "setuptools<82"
pip install transformers trl peft bitsandbytes datasets accelerate
```

## Cách chạy

### 1. Cấu hình Hugging Face token

Mở file script và thay thế:

```python
HF_TOKEN = "hf key của bạn"
```

bằng token thật của bạn, ví dụ:

```python
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 2. Chọn phiên bản mô hình

- Dùng `zenith:0.5b.py` cho mô hình 0.5B
- Dùng `zenith:1.5b.py` cho mô hình 1.5B / 1.8B

### 3. Chạy trong notebook hoặc máy local

```bash
python zenith:0.5b.py
```

hoặc:

```bash
python "zenith:1.5b.py"
```

Lưu ý: vì tên file có dấu `:`, nên trong một số môi trường bạn nên chạy qua notebook hoặc shell hỗ trợ tên file có ký tự đặc biệt.

## Lưu ý quan trọng

- Đây là dự án tập trung vào nghiên cứu và fine-tuning mô hình AI
- Bạn cần lưu ý về tài nguyên GPU và dung lượng RAM/VRAM
- Một số mẩu dữ liệu hội thoại trong script được định nghĩa sẵn để tạo training examples
- Token của Hugging Face cần được giữ bí mật, không nên commit trực tiếp vào repo công khai

## Kết quả sau khi huấn luyện

Sau khi `trainer.train()` hoàn tất, script sẽ lưu mô hình vào thư mục:

- `./results_qwen_local` cho file `zenith:0.5b.py`
- `./results_zenith_local` cho file `zenith:1.5b.py`

## Ghi chú

Dự án này là dạng demo / research training repo để xây dựng chatbot đa ngôn ngữ dựa trên Qwen. Nó phù hợp cho người mới bắt đầu học về fine-tuning, LoRA, chat template và deployment mô hình ngôn ngữ.

## License

Repo hiện chưa có license rõ ràng. Nếu bạn định chia sẻ công khai hoặc dùng cho mục đích thương mại, nên thêm giấy phép phù hợp như MIT hoặc Apache 2.0.

## Liên hệ / góp ý

Nếu bạn muốn mở rộng dự án, có thể: 

- thêm dữ liệu huấn luyện mới
- cải thiện prompt và chat template
- tối ưu hóa hyperparameters
- thêm script inference để test model trực tiếp

---

Made for Zenith VN by the Zenith Mode project.
