# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
from vllm.logger import init_logger
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


@dataclass
class LoadRecoveryExampleConnectorMetadata(ExampleConnectorMetadata):
    req_to_block_ids: dict[str, set[int]] = field(default_factory=dict)

    @classmethod
    def from_base(cls, base: ExampleConnectorMetadata):
        return cls(requests=base.requests)


class LoadRecoveryExampleConnector(ExampleConnector):
    def __init__(self, vllm_config: "VllmConfig", 
                 role: KVConnectorRole,
                 kv_cache_config: Optional["KVCacheConfig"] = None,):
        logger.info(f"sykdebug: begin init LoadRecoveryExampleConnector")
        super().__init__(vllm_config=vllm_config, 
                         role=role,            
                         kv_cache_config=kv_cache_config,)
        self._async_load = vllm_config.kv_transfer_config.get_from_extra_config(
            "async_load", False
        )
        self._invalid_block_ids: set = None
        self._seen_requests: set = set()
        self._req_to_block_ids: dict[str, list[int]] = dict()
        logger.info(f"sykdebug: after init LoadRecoveryExampleConnector, _async_load={self._async_load}")

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, LoadRecoveryExampleConnectorMetadata)
        # sykdebug: 表示处理的第一个load请求
        index, failed_request = next(
            (
                (i, x)
                for i, x in enumerate(connector_metadata.requests)
                if not x.is_store
            ),
            (None, None),
        )
        logger.info(f"sykdebug: during bind_connector_metadata, index={index}, failed_request, delete it from meta data")
        
        for _index, _request in enumerate(connector_metadata.requests):
            logger.info(f"sykdebug: during bind_connector_metadata, _index={_index}, _request.slot_mapping={_request.slot_mapping}")
        if index is not None:
            del connector_metadata.requests[index]
            self._invalid_block_ids = set(
                (
                    # sykdebug: 对应token涵盖的block_id
                    failed_request.slot_mapping[:: self._block_size] // self._block_size
                ).tolist()
            )
            logger.info(f"sykdebug: during bind_connector_metadata, _invalid_block_ids={self._invalid_block_ids}")
            logger.info(
                "Simulating failure to load all KV blocks for the "
                "first load request. Total blocks: %d",
                len(self._invalid_block_ids),
            )
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self._invalid_block_ids = None
        super().clear_connector_metadata()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        if self._async_load and forward_context.attn_metadata is None:
            # Bypass  sanity check in super().start_load_kv
            forward_context.attn_metadata = "None"
        logger.info(f"sykdebug: during start_load_kv, async_load={self._async_load}, forward_context.attn_metadata={forward_context.attn_metadata}")

        super().start_load_kv(forward_context, **kwargs)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self._async_load:
            meta = self._get_connector_metadata()
            assert isinstance(meta, LoadRecoveryExampleConnectorMetadata)
            if meta.req_to_block_ids:
                return None, set(meta.req_to_block_ids)

        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        logger.info(f"sykdebug: during get_blocks_ids_with_load_errors, _invalid_block_ids={self._invalid_block_ids}")
        return self._invalid_block_ids

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        
        # sykdebug: 检测请求是否被回退（num_computed_tokens 减少）
        # 如果被回退，需要从 _seen_requests 中移除，以便重新触发 KV 加载
        if request.request_id in self._seen_requests:
            logger.info(f"sykdebug: during get_num_new_matched_tokens, request.request_id={request.request_id} never been seen")
            return 0, False

        logger.info(f"sykdebug: during get_num_new_matched_tokens, request.request_id={request.request_id} add to seen_requests")
        self._seen_requests.add(request.request_id)

        num_tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        return num_tokens, self._async_load and num_tokens > 0

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        super().update_state_after_alloc(request, blocks, num_external_tokens)

        if num_external_tokens > 0:
            logger.info(f"sykdebug: during update_state_after_alloc, "
                        f"request.request_id={request.request_id}, "
                        f"blocks.get_block_ids()[0]={blocks.get_block_ids()[0]}, set to _req_to_block_ids")
            # sykdebug: 记录每个request覆盖的block_ids
            self._req_to_block_ids[request.request_id] = blocks.get_block_ids()[0]

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        if not self._async_load:
            base = super().build_connector_meta(scheduler_output)
            meta = LoadRecoveryExampleConnectorMetadata.from_base(base)
        else:
            meta = LoadRecoveryExampleConnectorMetadata()
            if self._requests_need_load:
                for req_id, request in self._requests_need_load.items():
                    logger.info(f"sykdebug: during build_connector_meta, async load, "
                                f"for req_id={req_id}, len(requests.token_ids)={len(request.all_token_ids)}")
                    meta.add_request(
                        token_ids=request.prompt_token_ids,
                        block_ids=self._req_to_block_ids[req_id],
                        block_size=self._block_size,
                        is_store=False,
                        mm_hashes=[],
                    )
                # Clear state
                self._requests_need_load.clear()
        meta.req_to_block_ids = self._req_to_block_ids
        self._req_to_block_ids = dict()
        return meta
