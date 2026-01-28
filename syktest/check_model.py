#!/usr/bin/env python3
"""
vLLM 多GPU推理服务 - 最大化利用2张GPU，支持压测和性能监控
"""
from vllm import LLM, SamplingParams
import torch
import time
import logging
from datetime import datetime
from typing import List, Dict
import sys

# 配置
model = "Qwen/Qwen3-0.6B"
# model = "Qwen/Qwen1.5-14B-Chat"
# model =  "Qwen/Qwen2.5-14B-Instruct"
# model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# model = "openbmb/MiniCPM4-8B"
MODEL_PATH = f"/root/shiyukun/models/{model}"  # 统一模型保存目录
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.5
TENSOR_PARALLEL_SIZE = 2  # 使用2张GPU
LOG_FILE = "tmp.log"

# 配置日志系统
def setup_logging():
    """配置日志，同时输出到文件和控制台"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 文件处理器 - 添加立即刷新
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 强制立即刷新缓冲区
    file_handler.flush()
    
    return logger

logger = setup_logging()

class VLLMService:
    """vLLM推理服务"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """初始化vLLM服务"""
        self.model_load_start = time.time()
        logger.info("=" * 70)
        logger.info("开始加载vLLM服务")
        logger.info(f"模型路径: {model_path}")
        logger.info(f"张量并行: {TENSOR_PARALLEL_SIZE} GPU")
        logger.info(f"GPU内存利用率: {GPU_MEMORY_UTILIZATION}")
        logger.info(f"最大模型长度: {MAX_MODEL_LEN}")
        logger.info("=" * 70)
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        
        self.model_load_time = time.time() - self.model_load_start
        logger.info(f"✅ 模型加载完成！耗时: {self.model_load_time:.2f}秒")
        logger.info("=" * 70)
    
    def warmup(self, num_prompts: int = 3):
        """模型预热"""
        logger.info(f"开始预热模型 ({num_prompts}个请求)...")
        warmup_start = time.time()
        warmup_prompts = ["你好" for _ in range(num_prompts)]
        self.generate(warmup_prompts, max_tokens=32, temperature=0.0)
        warmup_time = time.time() - warmup_start
        logger.info(f"✅ 预热完成！耗时: {warmup_time:.2f}秒")
        logger.info("=" * 70)
        return warmup_time
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成文本"""
        sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def generate_with_stats(self, prompts: List[str], **kwargs) -> Dict:
        """带性能统计的文本生成"""
        start_time = time.time()
        outputs = self.llm.generate(prompts, SamplingParams(**kwargs))
        end_time = time.time()
        
        total_time = end_time - start_time
        total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        
        return {
            "responses": [output.outputs[0].text for output in outputs],
            "stats": {
                "total_time": total_time,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "tokens_per_second": total_output_tokens / total_time if total_time > 0 else 0,
                "requests": len(prompts)
            }
        }
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话生成"""
        sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.chat([messages], sampling_params)
        return outputs[0].outputs[0].text

def run_benchmark(service: VLLMService, config: Dict):
    """运行压测"""
    logger.info("=" * 70)
    logger.info("开始基准测试")
    logger.info("=" * 70)
    logger.info(f"配置: 并发={config['concurrency']}, 请求数={config['requests']}, max_tokens={config['max_tokens']}")
    logger.info("=" * 70)
    
    # 准备测试prompt
    test_prompt = "请详细介绍人工智能的发展历程，包括关键里程碑和技术突破。"
    prompts = [test_prompt] * config['requests']
    
    # 预热
    service.warmup(num_prompts=2)
    
    # 分批执行
    batch_size = config['concurrency']
    total_stats = {
        "total_time": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_requests": 0
    }
    
    benchmark_start = time.time()
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        logger.info(f"执行批次: {i+1}-{min(i+batch_size, len(prompts))}/{len(prompts)}")
        
        result = service.generate_with_stats(
            batch,
            max_tokens=config['max_tokens'],
            temperature=0.7,
            top_p=0.95
        )
        
        stats = result['stats']
        total_stats['total_time'] += stats['total_time']
        total_stats['total_input_tokens'] += stats['total_input_tokens']
        total_stats['total_output_tokens'] += stats['total_output_tokens']
        total_stats['total_requests'] += stats['requests']
        
        # 显示本批次结果
        logger.info(f"  本批次: 耗时={stats['total_time']:.2f}s, "
                   f"tokens={stats['total_output_tokens']}, "
                   f"速度={stats['tokens_per_second']:.2f} tokens/s")
    
    total_benchmark_time = time.time() - benchmark_start
    
    # 输出汇总统计
    logger.info("=" * 70)
    logger.info("基准测试结果汇总")
    logger.info("=" * 70)
    logger.info(f"总请求数: {total_stats['total_requests']}")
    logger.info(f"总耗时: {total_benchmark_time:.2f}秒")
    logger.info(f"输入tokens: {total_stats['total_input_tokens']}")
    logger.info(f"输出tokens: {total_stats['total_output_tokens']}")
    logger.info(f"平均每请求耗时: {total_benchmark_time / total_stats['total_requests']:.2f}秒")
    logger.info(f"吞吐量: {total_stats['total_requests'] / total_benchmark_time:.2f} requests/s")
    logger.info(f"生成速度: {total_stats['total_output_tokens'] / total_benchmark_time:.2f} tokens/s")
    logger.info(f"模型加载耗时: {service.model_load_time:.2f}秒")
    logger.info("=" * 70)
    
    return total_stats

def demo():
    """演示推理功能"""
    logger.info("=== 示例演示 ===")
    service = VLLMService()
    service.warmup()
    
    # 示例1: 文本生成
    logger.info("\n示例1: 文本生成")
    prompts = ["请介绍一下人工智能的发展历程。", "Python的优点是什么？"]
    results = service.generate(prompts, temperature=0.7, max_tokens=256)
    for prompt, result in zip(prompts, results):
        logger.info(f"问题: {prompt}")
        logger.info(f"回答: {result[:200]}...")
    
    # 示例2: 对话模式
    logger.info("\n示例2: 对话模式")
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "如何学好编程？"}
    ]
    response = service.chat(messages, temperature=0.8, max_tokens=256)
    logger.info(f"用户: {messages[-1]['content']}")
    logger.info(f"助手: {response[:200]}...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='vLLM推理服务压测')
    parser.add_argument('--mode', choices=['demo', 'benchmark'], default='demo',
                        help='运行模式: demo(演示) 或 benchmark(压测)')
    parser.add_argument('--concurrency', type=int, default=4,
                        help='压测并发数')
    parser.add_argument('--requests', type=int, default=20,
                        help='压测总请求数')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='生成最大tokens数')
    
    args = parser.parse_args()
    
    logger.info(f"启动模式: {args.mode}")
    logger.info(f"日志文件: {LOG_FILE}")
    
    if args.mode == 'demo':
        demo()
    else:
        service = VLLMService()
        run_benchmark(service, {
            'concurrency': args.concurrency,
            'requests': args.requests,
            'max_tokens': args.max_tokens
        })
    
    logger.info("✅ 程序执行完成！")
    
    # 确保所有日志都写入文件
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()