from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent_runtime.workflow import MultiAgentWorkflow


def parse_resume_payload(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}") and ":" in stripped:
            body = stripped[1:-1]
            key, value = body.split(":", 1)
            key = key.strip().strip("\"'")
            value = value.strip().strip("\"'")
            if key == "action":
                return {"action": value}
        return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TS-Explain multi-agent workflow.")
    parser.add_argument("question", nargs="?", help="Question to answer.")
    parser.add_argument("--question", dest="question_flag", default=None)
    parser.add_argument("--provider", default="offline", choices=["offline", "qwen", "deepseek"])
    parser.add_argument("--dataset", dest="dataset_name", default=None)
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--auto-window", action="store_true", help="Use real-window analyzer when routing allows it.")
    parser.add_argument("--window-length", type=int, default=256)
    parser.add_argument("--window-strategy", default="max_label_density", choices=["max_label_density", "first_anomaly"])
    parser.add_argument("--runs-dir", default=None, help="Optional output directory for agent run records.")
    parser.add_argument("--engine", default="classic", choices=["classic", "langgraph"], help="Workflow engine to run.")
    parser.add_argument("--human-review", action="store_true", help="Pause LangGraph before answer generation.")
    parser.add_argument("--thread-id", default=None, help="LangGraph checkpoint thread id.")
    parser.add_argument("--resume-json", default=None, help="Resume a LangGraph interrupt with this JSON payload.")
    parser.add_argument("--checkpoint-backend", default=None, choices=["memory", "sqlite"])
    parser.add_argument("--checkpoint-path", default=None, help="SQLite checkpoint database path.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON instead of answer text.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    question = args.question_flag or args.question
    if args.resume_json is not None:
        if args.engine != "langgraph":
            raise SystemExit("--resume-json is only supported with --engine langgraph")
        if not args.thread_id:
            raise SystemExit("--resume-json requires --thread-id")
        checkpoint_backend = args.checkpoint_backend or "sqlite"
        if checkpoint_backend == "memory":
            raise SystemExit(
                "--resume-json requires a persistent checkpoint backend. "
                "Use --checkpoint-backend sqlite --checkpoint-path <path-to-same-db>."
            )
        from agent_runtime.langgraph_workflow import LangGraphWorkflow

        resume_payload = parse_resume_payload(args.resume_json)
        workflow = LangGraphWorkflow(
            provider=args.provider,
            runs_dir=args.runs_dir,
            checkpoint_backend=checkpoint_backend,
            checkpoint_path=args.checkpoint_path,
        )
        result = workflow.resume_interrupt(args.thread_id, resume_payload)
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return
        print(f"thread_id={result.thread_id}")
        print(f"status={result.state.status}")
        print("")
        print(result.state.answer or "")
        print("")
        print(f"records={result.output_dir}")
        return

    if not question:
        raise SystemExit('Please provide a question, e.g. python -m agent_runtime.cli "Why did DADA work on SMD?"')
    if args.human_review and args.engine != "langgraph":
        raise SystemExit("--human-review is only supported with --engine langgraph")
    if args.engine == "langgraph":
        from agent_runtime.langgraph_workflow import LangGraphWorkflow

        checkpoint_backend = args.checkpoint_backend or ("sqlite" if args.human_review else "memory")
        workflow = LangGraphWorkflow(
            provider=args.provider,
            runs_dir=args.runs_dir,
            checkpoint_backend=checkpoint_backend,
            checkpoint_path=args.checkpoint_path,
        )
        if args.human_review:
            result = workflow.run_interruptible(
                question=question,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                auto_window=args.auto_window,
                window_length=args.window_length,
                window_strategy=args.window_strategy,
                human_review=True,
                thread_id=args.thread_id,
            )
            if args.json:
                print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
                return
            print(f"thread_id={result.thread_id}")
            print(f"status={result.state.status}")
            if result.interrupted:
                print("interrupted=true")
                print(json.dumps(result.interrupts, ensure_ascii=False, indent=2))
                print(f"checkpoint_backend={checkpoint_backend}")
                if workflow.checkpoint_path:
                    print(f"checkpoint_path={workflow.checkpoint_path}")
                if checkpoint_backend == "memory":
                    print("note=memory checkpoints can resume only inside the same Python process.")
                print(f"records={result.output_dir}")
                return
            print("")
            print(result.state.answer or "")
            print("")
            print(f"records={result.output_dir}")
            return
    else:
        workflow = MultiAgentWorkflow(provider=args.provider, runs_dir=args.runs_dir)
    state = workflow.run(
        question=question,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        auto_window=args.auto_window,
        window_length=args.window_length,
        window_strategy=args.window_strategy,
    )
    if args.json:
        print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    print(f"run_id={state.run_id}")
    print(f"status={state.status}")
    if state.memory_summary:
        print(f"memory={state.memory_summary.summary}")
    print("")
    print(state.answer or "")
    print("")
    print(f"records={workflow.store.run_dir(state.run_id)}")


if __name__ == "__main__":
    main()
