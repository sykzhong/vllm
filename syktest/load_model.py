from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 配置国内镜像站
HF_MIRROR = "https://hf-mirror.com"  # HuggingFace 国内镜像

# 模型保存目录
MODEL_SAVE_DIR = "/root/shiyukun/models"  # 统一保存所有模型

# 可选：设置 HuggingFace Token（用于下载受限制的模型，如 Llama-3.1）
# 获取方式：访问 https://huggingface.co/settings/tokens 生成 Read Token
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxx"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # 从环境变量读取，或在此处直接填写

# 创建模型保存目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"模型保存目录: {MODEL_SAVE_DIR}")

# 设置镜像站环境变量
if HF_MIRROR:
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"使用镜像站: {HF_MIRROR}")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print(f"已配置 HF Token: {HF_TOKEN[:10]}...")

# 模型列表（可以注释掉不需要下载的模型）
models = [
    "Qwen/Qwen3-0.6B", # ✅ 开源，无需 token
    "openai/gpt-oss-20b",  # ✅ Apache 2.0 开源，无需 token
    "Qwen/Qwen2.5-14B-Instruct",  # ✅ 开源，无需 token
    "Qwen/Qwen1.5-14B-Chat",    # ✅ 开源，无需 token
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # ✅ 开源，无需 token
    # "meta-llama/Llama-3.1-8B-Instruct",  # ❌ 需要 token（已注释）
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",  # ⚠️ 可能需要 token
    "openbmb/MiniCPM4-8B",  # ✅ 开源，无需 token
    "deepseek-ai/DeepSeek-V2-Lite",  # ✅ 开源，无需 token
    # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",  # ✅ 开源，无需 token
    # "Qwen/Qwen3-14B-AWQ",  # ✅ Apache 2.0 开源，无需 token（AWQ量化版）
]


def load_model(model_name):
    MODEL_NAME=model_name
    # 构建本地保存路径：将模型名映射到统一的 models 目录
    local_dir = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    
    print(f"\n开始下载模型: {MODEL_NAME}")
    print(f"保存路径: {local_dir}")
    
    try:
        # 方案1: 使用 snapshot_download 仅下载文件不加载到GPU
        try:
            from huggingface_hub import snapshot_download
            # 一次性下载所有文件，不加载到GPU
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=local_dir,
                token=os.environ.get("HF_TOKEN")
            )
            print(f'✅ {MODEL_NAME} 下载完成（不加载到GPU）')
            
        except ImportError:
            # 方案2: 备选方案 - 仅下载分词器，不加载模型
            os.makedirs(local_dir, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            tokenizer.save_pretrained(local_dir)
            print(f'✅ {MODEL_NAME} 分词器下载完成')
            
    except Exception as e:
        print(f'❌ {MODEL_NAME} 下载失败: {str(e)}')
        print(f'   提示：如果是 "gated repo" 错误，请访问 HuggingFace 申请权限并配置 HF_TOKEN')
        print(f'   如果是 "CUDA out of memory" 错误，请检查是否有其他进程占用显存')
    

if __name__ == "__main__":
    for model in models:
        load_model(model)