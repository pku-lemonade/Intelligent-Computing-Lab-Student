"""Print the Week 01 worksheet calculations from a public JSON trace."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("trace.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = [item for item in data["requests"] if item["finished_at"] is not None]
    for item in data["requests"]:
        tokens = item["token_timestamps"]
        itl = [right - left for left, right in zip(tokens, tokens[1:])]
        tpot = sum(itl) / len(itl) if itl else None
        e2e = item["finished_at"] - item["arrived_at"] if item["finished_at"] is not None else None
        print(item["id"], {"ttft": tokens[0] - item["arrived_at"], "itl": itl, "tpot": tpot, "e2e": e2e})
    duration = data["window"]["end"] - data["window"]["start"] if "end" in data["window"] else data["window"]["stop"] - data["window"]["start"]
    output_tokens = sum(len(item["token_timestamps"]) for item in completed)
    print({"completed": len(completed), "output_tokens": output_tokens, "requests_per_s": len(completed) / duration, "output_tokens_per_s": output_tokens / duration})


if __name__ == "__main__":
    main()
