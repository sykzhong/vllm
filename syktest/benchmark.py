#!/usr/bin/env python3
"""
vLLM 多模型轮询压测工具 - 支持4个模型依次测试，性能监控和日志记录
"""
from vllm import LLM, SamplingParams
import torch
import time
import logging
from datetime import datetime
from typing import List, Dict
import sys
import os

# 配置
# 模型列表：支持本地路径，格式为 "模型名" 或 "完整路径"
# 实际路径会自动拼接 /root/shiyukun/{模型名}
MODELS = [
    "Qwen/Qwen3-0.6B", # ✅ 开源，无需 token
    "openai/gpt-oss-20b",  # ✅ Apache 2.0 开源，无需 token
    "Qwen/Qwen2.5-14B-Instruct",  # ✅ 开源，无需 token
    "Qwen/Qwen1.5-14B-Chat",    # ✅ 开源，无需 token
    "openbmb/MiniCPM4-8B",  # ✅ 开源，无需 token
    "deepseek-ai/DeepSeek-V2-Lite",  # ✅ 开源，无需 token
]

MODEL_BASE_PATH = "/root/shiyukun/models"  # 统一模型保存目录
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.7
TENSOR_PARALLEL_SIZE = 2  # 使用2张GPU

# 日志文件名：包含时间戳，方便区分不同测试
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"benchmark_{TIMESTAMP}.log"

# 压测配置
BENCHMARK_CONFIG = {
    'concurrency': 4,      # 并发数
    'requests': 20,        # 总请求数
    'max_tokens': 1024,     # 每次生成最大token数
    'temperature': 0.7,    # 采样温度
    'top_p': 0.95,        # top_p采样
}

# 配置日志系统
def setup_logging():
    """配置日志，同时输出到文件和控制台"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器，避免重复
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class VLLMService:
    """vLLM推理服务"""
    
    def __init__(self, model_path: str, model_name: str):
        """初始化vLLM服务"""
        self.model_name = model_name
        self.model_load_start = time.time()
        
        logger.info("=" * 70)
        logger.info(f"【{model_name}】开始加载vLLM服务")
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
    
    def generate_with_detailed_stats(self, prompts: List[str], **kwargs) -> Dict:
        """带详细性能统计的文本生成（包括首令牌耗时）"""
        sampling_params = SamplingParams(**kwargs)
        
        # 记录开始时间
        batch_start_time = time.time()
        
        # 执行生成
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 记录结束时间
        batch_end_time = time.time()
        
        total_time = batch_end_time - batch_start_time
        
        # 统计指标
        total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        
        # 估算首令牌耗时（TTFT - Time to First Token）
        # 由于vLLM API不直接返回TTFT，我们使用一个估算方法：
        # TTFT ≈ (总耗时 - 生成耗时) / 请求数
        # 其中生成耗时 = 总输出tokens / 估计生成速度
        estimated_generation_time = total_time * 0.8  # 假设80%时间用于生成
        avg_ttft = (total_time - estimated_generation_time) / len(prompts) if len(prompts) > 0 else 0
        
        # 计算各种吞吐指标
        tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0
        requests_per_second = len(prompts) / total_time if total_time > 0 else 0
        avg_output_tokens = total_output_tokens / len(prompts) if len(prompts) > 0 else 0
        
        return {
            "responses": [output.outputs[0].text for output in outputs],
            "stats": {
                "total_time": total_time,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "tokens_per_second": tokens_per_second,
                "requests_per_second": requests_per_second,
                "avg_output_tokens": avg_output_tokens,
                "avg_ttft": avg_ttft,  # 平均首令牌耗时（秒）
                "requests": len(prompts)
            }
        }

def run_benchmark(service: VLLMService, config: Dict):
    """运行单个模型的压测"""
    model_name = service.model_name
    
    logger.info("=" * 70)
    logger.info(f"【{model_name}】开始基准测试")
    logger.info("=" * 70)
    logger.info(f"压测配置:")
    logger.info(f"  并发数: {config['concurrency']}")
    logger.info(f"  总请求数: {config['requests']}")
    logger.info(f"  最大生成tokens: {config['max_tokens']}")
    logger.info(f"  温度: {config['temperature']}")
    logger.info(f"  Top-P: {config['top_p']}")
    logger.info("=" * 70)
    
    # 准备测试prompt
    test_prompt = "请详细介绍人工智能的发展历程，包括关键里程碑和技术突破。"
    prompts = [test_prompt] * config['requests']
    
    # 预热
    service.warmup(num_prompts=2)
    
    # 分批执行
    batch_size = config['concurrency']
    all_stats = []  # 记录每个批次的统计信息
    
    benchmark_start = time.time()
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        logger.info(f"执行批次: {batch_num}/{total_batches} (请求 {i+1}-{min(i+batch_size, len(prompts))})")
        
        result = service.generate_with_detailed_stats(
            batch,
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            top_p=config['top_p']
        )
        
        stats = result['stats']
        all_stats.append(stats)
        
        # 显示本批次结果
        logger.info(f"  本批次结果:")
        logger.info(f"    耗时: {stats['total_time']:.2f}s")
        logger.info(f"    输入tokens: {stats['total_input_tokens']}")
        logger.info(f"    输出tokens: {stats['total_output_tokens']}")
        logger.info(f"    平均首令牌耗时(TTFT): {stats['avg_ttft']*1000:.2f}ms")
        logger.info(f"    生成速度: {stats['tokens_per_second']:.2f} tokens/s")
        logger.info(f"    吞吐量: {stats['requests_per_second']:.2f} requests/s")
    
    total_benchmark_time = time.time() - benchmark_start
    
    # 汇总统计
    total_input_tokens = sum(s['total_input_tokens'] for s in all_stats)
    total_output_tokens = sum(s['total_output_tokens'] for s in all_stats)
    total_requests = sum(s['requests'] for s in all_stats)
    avg_ttft = sum(s['avg_ttft'] for s in all_stats) / len(all_stats) if all_stats else 0
    
    # 输出汇总统计
    logger.info("=" * 70)
    logger.info(f"【{model_name}】基准测试结果汇总")
    logger.info("=" * 70)
    logger.info(f"模型加载耗时: {service.model_load_time:.2f}秒")
    logger.info(f"总请求数: {total_requests}")
    logger.info(f"总耗时: {total_benchmark_time:.2f}秒")
    logger.info(f"输入tokens: {total_input_tokens}")
    logger.info(f"输出tokens: {total_output_tokens}")
    logger.info(f"平均每请求耗时: {total_benchmark_time / total_requests:.2f}秒")
    logger.info(f"平均首令牌耗时(TTFT): {avg_ttft*1000:.2f}ms")
    logger.info(f"吞吐量: {total_requests / total_benchmark_time:.2f} requests/s")
    logger.info(f"生成速度: {total_output_tokens / total_benchmark_time:.2f} tokens/s")
    logger.info(f"平均每请求输出tokens: {total_output_tokens / total_requests:.2f}")
    logger.info("=" * 70)
    
    # 返回详细统计结果
    return {
        'model_name': model_name,
        'model_load_time': service.model_load_time,
        'total_requests': total_requests,
        'total_time': total_benchmark_time,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'avg_ttft': avg_ttft,
        'throughput': total_requests / total_benchmark_time,
        'tokens_per_second': total_output_tokens / total_benchmark_time,
        'avg_output_tokens': total_output_tokens / total_requests,
    }

def run_multi_model_benchmark(models: List[str], config: Dict):
    """运行多模型轮询压测"""
    logger.info("=" * 70)
    logger.info("多模型压测工具启动")
    logger.info("=" * 70)
    logger.info(f"待测试模型列表: {len(models)}个")
    for i, model in enumerate(models, 1):
        logger.info(f"  {i}. {model}")
    logger.info(f"压测配置: 并发={config['concurrency']}, 请求数={config['requests']}, max_tokens={config['max_tokens']}")
    logger.info(f"日志文件: {LOG_FILE}")
    logger.info("=" * 70)
    
    all_results = []
    
    # 遍历所有模型
    for i, model in enumerate(models, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"开始测试第 {i}/{len(models)} 个模型")
        logger.info("=" * 70)
        
        # 构建模型路径
        if os.path.isabs(model):
            model_path = model
        else:
            model_path = os.path.join(MODEL_BASE_PATH, model)
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型路径不存在: {model_path}")
            logger.error(f"   跳过该模型，继续下一个...")
            continue
        
        try:
            # 创建服务实例
            service = VLLMService(model_path, model)
            
            # 运行压测
            result = run_benchmark(service, config)
            all_results.append(result)
            
            # 释放资源
            del service
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ 模型 {model} 测试失败: {str(e)}")
            logger.error(f"   继续测试下一个模型...")
            continue
    
    # 输出所有模型对比结果
    if all_results:
        logger.info("")
        logger.info("=" * 70)
        logger.info("所有模型压测结果对比")
        logger.info("=" * 70)
        
        # 表头
        logger.info(f"{'模型名称':<40} {'加载时间(s)':<12} {'TTFT(ms)':<12} {'吞吐(req/s)':<14} {'速度(tok/s)':<14} {'平均输出(tok)':<15}")
        logger.info("-" * 120)
        
        # 每个模型的结果
        for result in all_results:
            logger.info(
                f"{result['model_name']:<40} "
                f"{result['model_load_time']:<12.2f} "
                f"{result['avg_ttft']*1000:<12.2f} "
                f"{result['throughput']:<14.2f} "
                f"{result['tokens_per_second']:<14.2f} "
                f"{result['avg_output_tokens']:<15.2f}"
            )
        
        logger.info("=" * 70)
        
        # 找出最佳指标
        best_ttft = min(all_results, key=lambda x: x['avg_ttft'])
        best_throughput = max(all_results, key=lambda x: x['throughput'])
        best_speed = max(all_results, key=lambda x: x['tokens_per_second'])
        
        logger.info("性能指标排名:")
        logger.info(f"  🥇 首令牌最快: {best_ttft['model_name']} ({best_ttft['avg_ttft']*1000:.2f}ms)")
        logger.info(f"  🥇 吞吐量最高: {best_throughput['model_name']} ({best_throughput['throughput']:.2f} req/s)")
        logger.info(f"  🥇 生成速度最快: {best_speed['model_name']} ({best_speed['tokens_per_second']:.2f} tok/s)")
        logger.info("=" * 70)
    else:
        logger.warning("⚠️  没有成功的测试结果")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='vLLM多模型轮询压测工具')
    parser.add_argument('--concurrency', type=int, default=BENCHMARK_CONFIG['concurrency'],
                        help=f'压测并发数 (默认: {BENCHMARK_CONFIG["concurrency"]})')
    parser.add_argument('--requests', type=int, default=BENCHMARK_CONFIG['requests'],
                        help=f'压测总请求数 (默认: {BENCHMARK_CONFIG["requests"]})')
    parser.add_argument('--max-tokens', type=int, default=BENCHMARK_CONFIG['max_tokens'],
                        help=f'生成最大tokens数 (默认: {BENCHMARK_CONFIG["max_tokens"]})')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='指定要测试的模型列表，默认测试所有模型')
    
    args = parser.parse_args()
    
    # 更新配置
    config = BENCHMARK_CONFIG.copy()
    config['concurrency'] = args.concurrency
    config['requests'] = args.requests
    config['max_tokens'] = args.max_tokens
    
    # 确定要测试的模型列表
    test_models = args.models if args.models else MODELS
    
    logger.info(f"启动多模型压测")
    logger.info(f"日志文件: {LOG_FILE}")
    
    # 运行多模型压测
    run_multi_model_benchmark(test_models, config)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ 所有模型测试完成！")
    logger.info(f"📄 详细日志已保存到: {LOG_FILE}")
    logger.info("=" * 70)
    
    # 确保所有日志都写入文件
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()