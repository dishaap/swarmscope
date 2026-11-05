import random

def tamper(log, q, mode, rng):
    """Colluders delete their own coordination messages (each with probability q).
    cascade: they also delete their own replies to deleted messages, which
    removes the dangling references a naive tamper detector would look for."""
    dele = {m.msg_id for m in log if m.coord and rng.random() < q}
    if mode == "cascade":
        colluders = {m.sender for m in log if m.coord}
        changed = True
        while changed:
            changed = False
            for m in log:
                if m.msg_id not in dele and m.sender in colluders and m.reply_to in dele:
                    dele.add(m.msg_id); changed = True
    return [m for m in log if m.msg_id not in dele]

def degrade(log, completeness, q, mode, seed):
    """Platform log -> agent-side tampering -> monitor's partial view."""
    kept = tamper(log, q, mode, random.Random(f"{seed}|tamper|{q}|{mode}")) if q > 0 else list(log)
    view = random.Random(f"{seed}|view|{completeness}")
    return [m for m in kept if view.random() < completeness]