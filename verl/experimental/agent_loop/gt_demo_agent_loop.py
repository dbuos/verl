# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("gt_demo_agent")
class GTDemoAgentLoop(AgentLoopBase):
    """Demonstration agent loop: skips generation and returns the tokenized
    ground-truth markdown (extra_info["gt_markdown"]) as the response, so gold
    rows flow through the stock padding/position/reward/advantage pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. use the ground-truth markdown as the response instead of generating
        extra_info = kwargs.get("extra_info") or {}
        gold = extra_info.get("gt_markdown")
        if not gold:
            raise ValueError(f"[gt_demo] uid={kwargs.get('uid')} missing extra_info['gt_markdown']")

        response_ids = self.tokenizer.encode(gold, add_special_tokens=False)
        if len(response_ids) > self.response_length - 1:
            logger.warning(
                f"[gt_demo] truncating gold {len(response_ids)} -> {self.response_length - 1} tokens "
                f"uid={kwargs.get('uid')}"
            )
        response_ids = response_ids[: self.response_length - 1] + [self.tokenizer.eos_token_id]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0, num_preempted=-1),
            extra_fields={},
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
