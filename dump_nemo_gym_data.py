"""Dump Tau2/Tau3 rows for Nemo Gym.

The Gym prepare scripts consume the JSON files emitted by this script instead
of importing Tau internals directly.  This mirrors the original
``bxyu/nemo_gym_data`` branch shape while keeping the dump logic out of the
runtime simulation path.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.retrieval import get_all_variant_names
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.build import build_agent, build_orchestrator
from tau2.runner.helpers import get_tasks


OUTPUT_ROOT = Path("nemo_gym_data")


@dataclass(frozen=True)
class DumpSpec:
    name: str
    domain: str
    output_dir: str
    task_split_name: Optional[str] = "base"
    retrieval_config: Optional[str] = None


def _build_dump_specs() -> dict[str, DumpSpec]:
    specs = {
        "airline": DumpSpec(name="airline", domain="airline", output_dir="airline"),
        "retail": DumpSpec(name="retail", domain="retail", output_dir="retail"),
        "telecom": DumpSpec(name="telecom", domain="telecom", output_dir="telecom"),
    }
    for retrieval_config in get_all_variant_names():
        name = f"banking_{retrieval_config}"
        specs[name] = DumpSpec(
            name=name,
            domain="banking_knowledge",
            output_dir=f"banking_knowledge_{retrieval_config}",
            task_split_name=None,
            retrieval_config=retrieval_config,
        )
    return specs


DUMP_SPECS = _build_dump_specs()


def _responses_tools(environment: Environment) -> list[dict]:
    tools = []
    for tool in environment.get_tools():
        function_schema = tool.openai_schema["function"]
        tools.append({"type": "function", **function_schema, "strict": True})
    return tools


def _run_seed(config: TextRunConfig) -> int:
    return random.Random(config.seed).randint(0, 1000000)


def _make_config(spec: DumpSpec) -> TextRunConfig:
    kwargs = {
        "domain": spec.domain,
        "task_split_name": spec.task_split_name,
        "num_trials": 1,
        "llm_agent": "dummy",
        "llm_args_agent": {"api_base": "dummy", "api_key": "EMPTY"},
        "llm_user": "dummy",
    }
    if spec.retrieval_config is not None:
        kwargs["retrieval_config"] = spec.retrieval_config
    return TextRunConfig(**kwargs)


def _banking_environment(config: TextRunConfig, task: Task) -> Environment:
    """Build banking prompt/tools without embedding or sandbox side effects."""

    from tau2.domains.banking_knowledge.environment import get_db, get_knowledge_base
    from tau2.domains.banking_knowledge.retrieval import build_policy, resolve_variant
    from tau2.domains.banking_knowledge.retrieval_toolkits import (
        KnowledgeToolsAllTools,
        KnowledgeToolsPlain,
        KnowledgeToolsWithGrep,
        KnowledgeToolsWithKBSearch,
        KnowledgeToolsWithKBSearchAndGrep,
        KnowledgeToolsWithShell,
    )
    from tau2.domains.banking_knowledge.tools import KnowledgeUserTools

    db = get_db()
    knowledge_base = get_knowledge_base()
    variant = resolve_variant(
        config.retrieval_config or "alltools",
        **(config.retrieval_config_kwargs or {}),
    )
    policy = build_policy(variant, knowledge_base, task)

    has_all_tools = (
        variant.kb_search_bm25 is not None
        and variant.kb_search_dense is not None
        and variant.shell is not None
    )
    has_kb = variant.kb_search is not None
    has_grep = variant.grep is not None
    has_shell = variant.shell is not None

    if has_all_tools:
        tools = KnowledgeToolsAllTools(
            db,
            kb_bm25_pipeline=None,
            kb_dense_pipeline=None,
            sandbox=None,
        )
    elif has_shell:
        tools = KnowledgeToolsWithShell(db, sandbox=None)
    elif has_kb and has_grep:
        tools = KnowledgeToolsWithKBSearchAndGrep(
            db,
            kb_pipeline=None,
            grep_pipeline=None,
        )
    elif has_kb:
        tools = KnowledgeToolsWithKBSearch(db, kb_pipeline=None)
    elif has_grep:
        tools = KnowledgeToolsWithGrep(db, grep_pipeline=None)
    else:
        tools = KnowledgeToolsPlain(db)

    return Environment(
        domain_name="banking_knowledge",
        policy=policy,
        tools=tools,
        user_tools=KnowledgeUserTools(db),
    )


def _prompt_and_tools(config: TextRunConfig, task: Task) -> dict:
    if config.domain == "banking_knowledge":
        environment = _banking_environment(config, task)
        agent = build_agent(
            config.effective_agent,
            environment,
            llm=config.llm_agent,
            llm_args=config.llm_args_agent,
            task=task,
        )
    else:
        orchestrator = build_orchestrator(config, task, seed=_run_seed(config))
        environment = orchestrator.environment
        agent = orchestrator.agent

    return {
        "input": [
            {"role": "system", "content": message.content}
            for message in agent.get_init_state().system_messages
        ],
        "tools": _responses_tools(environment),
    }


def _row_for_task(spec: DumpSpec, config: TextRunConfig, task: Task) -> dict:
    return {
        "config": config.model_dump(mode="json"),
        "task": task.model_dump(mode="json"),
        "seed": _run_seed(config),
        "evaluation_type": EvaluationType.ALL.value,
        "save_dir": None,
        "user_voice_settings": None,
        "user_persona_config": None,
        "verbose_logs": False,
        "audio_debug": False,
        "audio_taps": False,
        "auto_review": False,
        "review_mode": "full",
        "hallucination_feedback": None,
        "responses_create_params": _prompt_and_tools(config, task),
    }


def dump_spec(spec: DumpSpec, output_root: Path) -> int:
    config = _make_config(spec)
    tasks = get_tasks(spec.domain, task_split_name=spec.task_split_name)
    if not tasks:
        raise RuntimeError(f"No tasks found for {spec.domain}")

    output_dir = output_root / spec.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        output_path = output_dir / f"{task.id}.json"
        output_path.write_text(
            json.dumps(_row_for_task(spec, config, task), indent=2) + "\n"
        )

    print(f"Wrote {len(tasks)} {spec.name} rows to {output_dir}")
    return len(tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DUMP_SPECS),
        help="Dataset to dump. May be provided more than once. Defaults to all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where nemo_gym_data-style JSON directories are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = args.dataset or list(DUMP_SPECS)
    total = 0
    for dataset_name in dataset_names:
        total += dump_spec(DUMP_SPECS[dataset_name], args.output_root)
    print(f"Wrote {total} total rows")


if __name__ == "__main__":
    main()
