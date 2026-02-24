# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Bug reproduction for: https://github.com/vllm-project/vllm/issues/34484
"""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import (
    create_model_runner_output,
    create_request,
    create_vllm_config,
)

logger = init_logger(__name__)

pytestmark = pytest.mark.cpu_test


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load: bool,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens

def create_async_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> AsyncScheduler:
    """Initialize Scheduler For Testing."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return AsyncScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
    )

@pytest.fixture
def recompute_scheduler():
    """scheduler with kv_load_failure_policy='recompute'"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_failure_policy = "recompute"
    return create_async_scheduler(vllm_config)


def test_kv_load_failure_bug_in_sync_mode(recompute_scheduler):
    """kv load failure bug repro"""
    
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * recompute_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * recompute_scheduler.block_size
    )
    num_valid_external_computed_tokens = num_external_computed_tokens - invalid_block_idx * recompute_scheduler.block_size
    num_uncomputed_tokens = num_prompt_tokens - num_external_computed_tokens

    request = create_request(num_tokens=num_prompt_tokens)
    recompute_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    # mock connector indicating sync load
    recompute_scheduler.connector = Mock()
    recompute_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    recompute_scheduler.connector.request_finished.return_value = (False, None)
    recompute_scheduler.connector.take_events.return_value = ()

    # First schedule: based on kv connector, complete the remaining num_uncomputed_tokens
    scheduler_output_1 = recompute_scheduler.schedule()

    # Verify scheduler_output_1: should complete num_uncomputed_tokens and add 1 placeholder
    assert len(recompute_scheduler.running) == 1
    assert scheduler_output_1.num_scheduled_tokens[request.request_id] == num_uncomputed_tokens
    assert request.num_computed_tokens == num_prompt_tokens
    assert request.num_output_placeholders == 1
    assert request.status == RequestStatus.RUNNING    
    assert request.num_tokens == 1600
    
    # Second schedule: in async scheduling mode, simulate scheduling before output is returned
    scheduler_output_2 = recompute_scheduler.schedule()
    
    # Verify scheduler_output_2: should add one output_token and one placeholder
    assert len(recompute_scheduler.running) == 1
    assert scheduler_output_2.num_scheduled_tokens[request.request_id] == 1
    assert request.num_computed_tokens == num_prompt_tokens + 1
    assert request.status == RequestStatus.RUNNING
    assert request.num_output_placeholders == 2
    assert request.num_tokens == 1600
    
    # get allocated invalid block IDs
    assert len(scheduler_output_1.scheduled_new_reqs) == 1
    req_block_ids = scheduler_output_1.scheduled_new_reqs[0].block_ids[0]
    invalid_block_id = req_block_ids[invalid_block_idx]
    invalid_block_ids = {invalid_block_id}
    
    # Return first schedule result with invalid_blocks
    model_runner_output_1 = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=False,
    )
    recompute_scheduler.update_from_output(scheduler_output_1, model_runner_output_1)

    # Update with first result: has invalid_blocks, num_output_placeholders won't be refreshed
    assert request.status == RequestStatus.RUNNING
    assert request.num_output_placeholders == 2
    assert request.num_tokens == 1600
    
    # Return second schedule result without invalid_block_ids
    model_runner_output_2 = create_model_runner_output(
        [request],
        invalid_block_ids=[],
        use_eos=False,
    )
    recompute_scheduler.update_from_output(scheduler_output_2, model_runner_output_2)

    # Second result has no invalid blocks, triggers refresh of num_output_placeholders and num_tokens
    assert request.status == RequestStatus.RUNNING
    assert request.num_output_placeholders == 1
    assert request.num_tokens == 1601
    
    # Third schedule: enter recompute phase
    scheduler_output_3 = recompute_scheduler.schedule()
    
    # Verify third schedule result: should re-prefill
    assert len(recompute_scheduler.running) == 1
    assert scheduler_output_3.num_scheduled_tokens[request.request_id] == recompute_scheduler.max_num_scheduled_tokens
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == num_valid_external_computed_tokens + num_uncomputed_tokens + recompute_scheduler.max_num_scheduled_tokens
    
    # Note: num_output_placeholders cannot be cleared, num_tokens > num_prompt_tokens !!
    assert request.num_output_placeholders == 1
    assert request.num_tokens == 1601    
    