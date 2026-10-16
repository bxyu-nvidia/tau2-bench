import asyncio
from copy import deepcopy

import pytest

from tau2.data_model.message import SystemMessage, ToolMessage, UserMessage
from tau2.utils import llm_utils


def _chat_response(content, *, reasoning_field=None, reasoning=None, tool_call=False):
    message = {"role": "assistant", "content": content}
    if reasoning_field is not None:
        message[reasoning_field] = reasoning
    if tool_call:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup_account", "arguments": '{"id": "42"}'},
            }
        ]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_call else "stop",
                "message": message,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _capture_model_requests(monkeypatch, responses):
    requests = []
    remaining_responses = iter(deepcopy(responses))

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def create_chat_completion(self, **kwargs):
            requests.append(kwargs)
            return next(remaining_responses)

    monkeypatch.setattr(llm_utils, "NeMoGymAsyncOpenAI", FakeClient)
    return requests


def _run_two_turns(*, replay_reasoning=False):
    replay_args = {"replay_reasoning": True} if replay_reasoning else {}
    history = [
        SystemMessage(role="system", content="Help the customer."),
        UserMessage(role="user", content="Check account 42."),
    ]
    first = asyncio.run(
        llm_utils.generate(
            "test-model",
            history,
            api_base="http://model.test/v1",
            api_key="test-key",
            **replay_args,
        )
    )
    history.extend(
        [first, ToolMessage(role="tool", id="call-1", content='{"balance": 5}')]
    )
    second = asyncio.run(
        llm_utils.generate(
            "test-model",
            history,
            api_base="http://model.test/v1",
            api_key="test-key",
            **replay_args,
        )
    )
    return first, second


@pytest.mark.parametrize(
    ("first_response", "expected_content"),
    [
        (_chat_response("<think>Look up account 42.</think>", tool_call=True), ""),
        (
            _chat_response(
                None,
                reasoning_field="reasoning_content",
                reasoning="Look up account 42.",
                tool_call=True,
            ),
            None,
        ),
        (
            _chat_response(
                None,
                reasoning_field="reasoning",
                reasoning="Look up account 42.",
                tool_call=True,
            ),
            None,
        ),
    ],
)
def test_opted_in_policy_replays_reasoning_after_tool_call(
    monkeypatch, first_response, expected_content
):
    requests = _capture_model_requests(
        monkeypatch, [first_response, _chat_response("The balance is 5.")]
    )

    first, second = _run_two_turns(replay_reasoning=True)

    assert first.content == expected_content
    assert first.tool_calls[0].id == "call-1"
    assert (
        first.raw_data["choices"][0]["message"].get("reasoning_content")
        or first.raw_data["choices"][0]["message"].get("reasoning")
    ) == "Look up account 42."
    assert second.content == "The balance is 5."
    assert requests[1]["messages"][2]["content"] == (
        "<think>Look up account 42.</think>"
    )
    assert requests[1]["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert requests[1]["messages"][3]["tool_call_id"] == "call-1"
    assert "reasoning_content" not in requests[1]["messages"][2]
    assert "replay_reasoning" not in requests[1]


def test_default_model_requests_do_not_replay_prior_reasoning(monkeypatch):
    requests = _capture_model_requests(
        monkeypatch,
        [
            _chat_response("<think>Look up account 42.</think>", tool_call=True),
            _chat_response("The balance is 5."),
        ],
    )

    first, second = _run_two_turns()

    assert first.tool_calls[0].id == "call-1"
    assert first.content == ""
    assert first.raw_data["choices"][0]["message"]["reasoning_content"] == ""
    assert second.content == "The balance is 5."
    assert "reasoning_content" not in requests[1]["messages"][2]
    assert requests[1]["messages"][2]["content"] == ""
