import json
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Msg:
    run_id: str
    msg_id: str         # board-assigned, monotonic: "m0001"
    seq: int            # per-sender counter: 0, 1, 2... (assumes platform stamps it)
    round: int
    sender: str
    channel: str
    content: str
    task_id: str
    reply_to: Optional[str] = None
    coord: bool = False # ground truth - monitors do not read

def save_log(path, log, meta):
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for m in log:
            f.write(json.dumps(asdict(m)) + "\n")

def load_log(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    return [Msg(**d) for d in lines[1:]], lines[0]["meta"]

