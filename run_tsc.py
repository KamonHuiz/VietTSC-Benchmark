#!/usr/bin/env python3
"""Train / evaluate traffic-signal controllers on the SUMO scenarios.

Examples
--------
  python run_tsc.py junctions                       # show agents, movements, phases
  python run_tsc.py eval  --algo fixedtime   --mode light            # test split by default
  python run_tsc.py eval  --algo maxpressure --mode heavy
  python run_tsc.py train --algo mplight     --mode medium --rounds 100   # trains on train, tests on val
  python run_tsc.py train --algo mplight     --mode medium --resume runs/mplight_medium_XXXX
  python run_tsc.py eval  --algo mplight     --mode heavy --model runs/.../models/round_99.pt

Splits (sumo/scenarios/<split>/<mode>/, fixed seeds in build_demand.py): train = 1001, val = 2002, test = 3003.
Training episodes use the train split, the greedy test every --test-every rounds uses the val split (pick the
checkpoint there), and final numbers are reported with `eval` on the test split. `--mode test-heavy` is the
original random stress test (sumo/scenarios/stress/test-heavy), identical for all splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tsc.agents import FixedTimeAgent, MaxPressureAgent, MPLightAgent  # noqa: E402
from tsc.config import ENV_CONF, EXP_CONF, MPLIGHT_AGENT_CONF, SUMO_CONF  # noqa: E402
from tsc.env import SumoTSCEnv  # noqa: E402
from tsc.junctions import build_junctions  # noqa: E402

SCENARIOS = ROOT / "sumo" / "scenarios"
SPLITS = ("train", "val", "test")


def scenario_cfg(split: str, mode: str) -> Path:
    if mode == "test-heavy":
        return SCENARIOS / "stress" / "test-heavy" / "osm.sumocfg"
    return SCENARIOS / split / mode / "osm.sumocfg"
NET_FILE = ROOT / "sumo" / "map" / "hcmc.net.xml.gz"
JUNCTION_FILE = ROOT / "sumo" / "config" / "junctions.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--mode", default="light", choices=("light", "medium", "heavy", "test-heavy"))
        sp.add_argument("--split", default=None, choices=SPLITS,
                        help="scenario split (default: train for `train`, test for `eval`)")
        sp.add_argument("--seed", type=int, default=SUMO_CONF["SIM_SEED"], help="SUMO seed")
        sp.add_argument("--gui", action="store_true", help="watch in sumo-gui (slow)")
        sp.add_argument("--traci", action="store_true", help="use TraCI instead of libsumo")
        sp.add_argument("--run-counts", type=int, default=ENV_CONF["RUN_COUNTS"], help="seconds per episode")
        sp.add_argument("--out", type=Path, default=ROOT / "runs")
        phase_mode(sp)

    def phase_mode(sp):
        sp.add_argument("--phase-mode", default=SUMO_CONF["PHASE_MODE"], choices=("plan", "conflict"),
                        help="plan: stages of the original signal plans (default); "
                             "conflict: maximal sets of compatible movements")

    phase_mode(sub.add_parser("junctions", help="print the junction agents built from the map"))

    tr = sub.add_parser("train", help="train MPLight")
    common(tr)
    tr.add_argument("--algo", default="mplight", choices=("mplight",))
    tr.add_argument("--rounds", type=int, default=EXP_CONF["NUM_ROUNDS"])
    tr.add_argument("--generators", type=int, default=EXP_CONF["NUM_GENERATORS"])
    tr.add_argument("--test-every", type=int, default=EXP_CONF["TEST_EVERY"],
                    help="greedy test episode every N rounds (0 = never)")
    tr.add_argument("--resume", type=Path, default=None, help="run folder to continue")
    tr.add_argument("--device", default="cpu")

    ev = sub.add_parser("eval", help="evaluate a controller")
    common(ev)
    ev.add_argument("--algo", required=True, choices=("mplight", "maxpressure", "fixedtime"))
    ev.add_argument("--model", type=Path, help="MPLight checkpoint (.pt)")
    ev.add_argument("--episodes", type=int, default=1, help="episodes with seeds seed, seed+1, ...")
    return p.parse_args()


def load_junctions(phase_mode=SUMO_CONF["PHASE_MODE"], log=print):
    conf = dict(SUMO_CONF, PHASE_MODE=phase_mode)
    junctions, skipped = build_junctions(NET_FILE, JUNCTION_FILE, conf, log=log)
    for jid, why in skipped:
        log(f"  junction {jid} not controlled: {why} (keeps its original plan)")
    return junctions


def make_env(args, junctions, control=True, split=None):
    env_conf = dict(ENV_CONF, RUN_COUNTS=args.run_counts)
    cfg = scenario_cfg(split or args.split, args.mode)
    return SumoTSCEnv(cfg, junctions, env_conf, seed=args.seed, gui=args.gui,
                      use_libsumo=not args.traci, control=control)


def run_episode(env, agent, explore=False, collect=False, seed=None):
    obs = env.reset(seed=seed)
    done, rewards = False, []
    while not done:
        actions = agent.act(obs, explore=explore)
        next_obs, rew, done = env.step(actions)
        if collect:
            agent.remember(obs, actions, rew, next_obs)
        rewards.append(sum(rew.values()) / max(len(rew), 1))
        obs = next_obs
    m = env.metrics()
    m["mean_reward"] = float(sum(rewards) / max(len(rewards), 1))
    env.close()
    return m


def fmt(m):
    return (f"ATT {m['travel_time']:.1f}s (arrived-only {m['travel_time_arrived']:.1f}s) | "
            f"arrived {m['arrived']}/{m['departed']} | queue {m['avg_queue']:.1f} | "
            f"teleports {m['teleports']} | reward {m['mean_reward']:.3f}")


# ---------------------------------------------------------------------------- commands

def cmd_junctions(args):
    for j in load_junctions(args.phase_mode):
        print(j.summary())
        print()


def cmd_eval(args):
    args.split = args.split or "test"
    junctions = load_junctions(args.phase_mode)
    if args.algo == "fixedtime":
        agent, env = FixedTimeAgent(), make_env(args, junctions, control=False)
    elif args.algo == "maxpressure":
        agent, env = MaxPressureAgent(junctions), make_env(args, junctions)
    else:
        if not args.model:
            sys.exit("--model is required for mplight evaluation")
        agent = MPLightAgent(junctions, MPLIGHT_AGENT_CONF)
        agent.load(args.model, with_optimizer=False)
        env = make_env(args, junctions)
    print(f"backend={env.backend} split={args.split} mode={args.mode} algo={args.algo}")
    results = []
    for k in range(args.episodes):
        t0 = time.time()
        m = run_episode(env, agent, explore=False, seed=args.seed + k)
        m.update(episode=k, seed=args.seed + k, wall_s=round(time.time() - t0, 1))
        results.append(m)
        print(f"[{args.algo}] episode {k}: {fmt(m)} ({m['wall_s']}s)")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"eval_{args.algo}_{args.split}_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


def cmd_train(args):
    args.split = args.split or "train"
    if args.resume:
        run_dir = args.resume
        saved = json.loads((run_dir / "config.json").read_text())
        args.mode, args.seed, args.run_counts = saved["mode"], saved["seed"], saved["run_counts"]
        args.split = saved.get("split", "train")
        args.phase_mode = saved.get("phase_mode", args.phase_mode)
    else:
        run_dir = args.out / f"mplight_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    junctions = load_junctions(args.phase_mode)
    agent = MPLightAgent(junctions, MPLIGHT_AGENT_CONF, seed=args.seed, device=args.device)
    env = make_env(args, junctions)
    val_env = make_env(args, junctions, split="val") if args.split == "train" else env

    start = 0
    if args.resume:
        rounds = sorted(int(p.stem.split("_")[1]) for p in model_dir.glob("round_*.pt"))
        if rounds:
            last = rounds[-1]
            agent.load(model_dir / f"round_{last}.pt")
            for r in rounds:                       # target-network snapshots
                if r >= last - MPLIGHT_AGENT_CONF["UPDATE_Q_BAR_FREQ"]:
                    agent.snapshots[r] = torch.load(model_dir / f"round_{r}.pt", map_location="cpu",
                                                    weights_only=False)["model"]
            mem_file = run_dir / "memory.pkl"
            if mem_file.is_file():
                agent.memory = pickle.loads(mem_file.read_bytes())
            start = last + 1
            print(f"resuming from round {last}")
    else:
        (run_dir / "config.json").write_text(json.dumps({
            "mode": args.mode, "split": args.split, "seed": args.seed, "run_counts": args.run_counts,
            "phase_mode": args.phase_mode,
            "rounds": args.rounds, "generators": args.generators,
            "agent": MPLIGHT_AGENT_CONF, "env": ENV_CONF, "sumo": SUMO_CONF,
            "junctions": {j.id: {"name": j.name, "tls": j.tls, "phases": j.num_phases,
                                 "movements": j.num_movements} for j in junctions},
        }, indent=2, ensure_ascii=False))

    log_file = run_dir / "progress.csv"
    fields = ["round", "epsilon", "train_travel_time", "train_reward", "loss", "val_loss", "epochs",
              "test_travel_time", "test_travel_time_arrived", "test_arrived", "test_queue",
              "test_teleports", "test_reward", "wall_s"]
    new_log = not log_file.is_file()
    print(f"run folder: {run_dir}\nbackend={env.backend} agents={len(junctions)} "
          f"max phases={agent.P} max movements={agent.M}")

    with log_file.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_log:
            writer.writeheader()
        for cnt_round in range(start, args.rounds):
            t0 = time.time()
            agent.set_round(cnt_round)
            train_ms = []
            for g in range(args.generators):
                train_ms.append(run_episode(env, agent, explore=True, collect=True,
                                            seed=args.seed + g))
            stats = agent.update(cnt_round)
            agent.save(model_dir / f"round_{cnt_round}.pt", cnt_round)
            (run_dir / "memory.pkl").write_bytes(pickle.dumps(agent.memory))

            row = {"round": cnt_round, "epsilon": round(agent.epsilon, 4),
                   "train_travel_time": round(sum(m["travel_time"] for m in train_ms) / len(train_ms), 2),
                   "train_reward": round(sum(m["mean_reward"] for m in train_ms) / len(train_ms), 4),
                   "loss": round(stats["loss"], 5), "val_loss": round(stats["val_loss"], 5),
                   "epochs": stats["epochs"]}
            msg = (f"round {cnt_round:3d} eps {agent.epsilon:.3f} | train ATT {row['train_travel_time']:.1f}s"
                   f" | loss {stats['loss']:.4f} val {stats['val_loss']:.4f} ({stats['epochs']} ep)")
            if args.test_every and cnt_round % args.test_every == 0:
                tm = run_episode(val_env, agent, explore=False, seed=args.seed)   # greedy test on val
                row.update(test_travel_time=round(tm["travel_time"], 2),
                           test_travel_time_arrived=round(tm["travel_time_arrived"], 2),
                           test_arrived=tm["arrived"], test_queue=round(tm["avg_queue"], 2),
                           test_teleports=tm["teleports"], test_reward=round(tm["mean_reward"], 4))
                msg += f" | test {fmt(tm)}"
            row["wall_s"] = round(time.time() - t0, 1)
            writer.writerow(row)
            fh.flush()
            print(msg + f" | {row['wall_s']}s", flush=True)


def main():
    args = parse_args()
    {"junctions": cmd_junctions, "eval": cmd_eval, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
