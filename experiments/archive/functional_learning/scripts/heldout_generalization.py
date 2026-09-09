#!/usr/bin/env python3
"""research: Held-out entity generalization test for focused masking.

Scientific purpose
------------------
The pilot showed focused masking closes the R-N gap on memorized pairs. The critical
question is whether the binding mechanism GENERALIZES to unseen entity-state combinations.
This directly parallels the synthetic held-symbol test: held-symbol transfer collapsed
under full-objective training but was preserved under answer-only training.

Design:
  - Generate 40 paired packets from templates with 20 different entity pairs
  - Split: 30 train pairs (60 packets), 10 held-out pairs (20 packets)
  - Held-out entities are DISJOINT from train entities
  - Train with focused masking on train pairs
  - Evaluate binding on both train and held-out pairs
  - Success = R-N gap closes on HELD-OUT pairs (not just train)

Usage
-----
  python heldout_generalization.py \\
    --model-path <coherent86_endpoint> \\
    --out-dir <output_directory> \\
    --gpu 0 --epochs 200
"""
import argparse, json, pathlib, sys, random, copy, time
from typing import Dict, List, Any
from collections import defaultdict

# ── Entity-state pools for template generation ──

TRAIN_ENTITY_PAIRS = [
    ("Alice", "Bob"), ("Professor Chen", "Dr. Reyes"),
    ("Tom", "Sarah"), ("Maria", "James"), ("Elena", "Marcus"),
    ("Chef Kim", "Chef Park"), ("David", "Sophia"), ("Rebecca", "Nathan"),
    ("Oliver", "Emma"), ("Lucas", "Mia"), ("Ethan", "Lily"),
    ("Dr. Foster", "Dr. Singh"), ("Captain Reed", "Captain Blake"),
    ("Maya", "Leo"), ("Grace", "Henry"),
]

HELD_ENTITY_PAIRS = [
    ("Daniel", "Rachel"), ("Mr. Thompson", "Ms. Garcia"),
    ("Felix", "Nora"), ("Professor Wright", "Professor Lin"),
    ("Samuel", "Clara"),
]

STATE_TEMPLATES = [
    {
        "attribute": "carrying",
        "source_template": "{E1} carried a {S1} and {E2} carried a {S2}.",
        "update_template": "Later, {UE} switched to carrying a {NS}.",
        "use_template": "When they arrived, {QE} was holding a {STATE}.",
        "states": [
            ("red umbrella", "blue bag", "green backpack", "yellow suitcase"),
            ("silver briefcase", "brown satchel", "black duffel", "white tote"),
            ("wool scarf", "cotton shawl", "silk tie", "linen handkerchief"),
        ],
    },
    {
        "attribute": "vehicle",
        "source_template": "{E1} drove a {S1} to work and {E2} drove a {S2}.",
        "update_template": "{UE} traded for a {NS} at the dealership.",
        "use_template": "In the parking lot, {QE} was loading groceries into a {STATE}.",
        "states": [
            ("silver sedan", "black SUV", "white truck", "red convertible"),
            ("blue minivan", "green hatchback", "gray pickup", "tan wagon"),
        ],
    },
    {
        "attribute": "pet",
        "source_template": "{E1} had a {S1} as a pet and {E2} had a {S2}.",
        "update_template": "{UE} adopted a {NS} from the shelter.",
        "use_template": "At the gathering, {QE} brought a {STATE} for everyone to see.",
        "states": [
            ("tabby cat", "golden retriever", "spotted rabbit", "gray parrot"),
            ("border collie", "Persian cat", "green turtle", "brown hamster"),
        ],
    },
    {
        "attribute": "instrument",
        "source_template": "{E1} played the {S1} and {E2} played the {S2}.",
        "update_template": "{UE} switched to the {NS} for the concert.",
        "use_template": "At the performance, {QE} impressed everyone with the {STATE}.",
        "states": [
            ("acoustic guitar", "grand piano", "electric violin", "steel drums"),
            ("classical flute", "jazz trumpet", "folk banjo", "rock bass"),
        ],
    },
    {
        "attribute": "workspace",
        "source_template": "{E1} kept {S1} on the desk and {E2} kept {S2} on the shelf.",
        "update_template": "{UE} replaced them with {NS} for the project.",
        "use_template": "A visitor noticed {QE} working with {STATE} at the office.",
        "states": [
            ("leather notebooks", "wooden pencils", "digital tablets", "printed charts"),
            ("glass beakers", "metal instruments", "plastic models", "paper diagrams"),
        ],
    },
]


def generate_pair_from_template(pair_id, e1, e2, template, state_set, rng):
    """Generate one paired packet from template."""
    s1, s2, ns1, ns2 = state_set
    
    source = template["source_template"].format(E1=e1, E2=e2, S1=s1, S2=s2)
    
    # UPDATE: e1 gets new state, ask about e1
    update_target = template["update_template"].format(UE=e1, NS=ns1)
    use_update = template["use_template"].format(QE=e1, STATE=ns1)
    
    # RETAIN: e2 gets new state, ask about e1  
    update_distractor = template["update_template"].format(UE=e2, NS=ns2)
    use_retain = template["use_template"].format(QE=e1, STATE=s1)
    
    use_frame = template["use_template"].replace("{STATE}", "{STATE}").replace("{QE}", e1)
    
    packets = [
        {
            "pair_id": pair_id,
            "packet_type": "UPDATE",
            "source_sentence": source,
            "update_sentence": update_target,
            "use_sentence": use_update,
            "use_sentence_frame": use_frame,
            "full_text": f"{source} {update_target} {use_update}",
            "answer_text": ns1,
            "foil_text": s1,
            "entity_name": e1,
            "update_entity": e1,
            "update_state_text": ns1,
            "source_state_text": s1,
        },
        {
            "pair_id": pair_id,
            "packet_type": "RETAIN",
            "source_sentence": source,
            "update_sentence": update_distractor,
            "use_sentence": use_retain,
            "use_sentence_frame": use_frame,
            "full_text": f"{source} {update_distractor} {use_retain}",
            "answer_text": s1,
            "foil_text": ns1,
            "entity_name": e1,
            "update_entity": e2,
            "update_state_text": ns2,
            "source_state_text": s1,
        },
    ]
    return packets


def generate_all_pairs(entity_pairs, prefix, rng):
    """Generate diverse pairs from entity pairs and templates."""
    all_packets = []
    pair_idx = 0
    
    for ep_idx, (e1, e2) in enumerate(entity_pairs):
        # Each entity pair gets 2 templates with different state sets
        template_choices = rng.sample(range(len(STATE_TEMPLATES)), min(2, len(STATE_TEMPLATES)))
        for tc_idx, ti in enumerate(template_choices):
            t = STATE_TEMPLATES[ti]
            state_set = rng.choice(t["states"])
            pair_id = f"{prefix}_{pair_idx:03d}"
            packets = generate_pair_from_template(pair_id, e1, e2, t, state_set, rng)
            all_packets.extend(packets)
            pair_idx += 1
    
    return all_packets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()
    
    import torch
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    rng = random.Random(43032)
    
    # Generate train and held-out pairs
    train_packets = generate_all_pairs(TRAIN_ENTITY_PAIRS, "tr", rng)
    held_packets = generate_all_pairs(HELD_ENTITY_PAIRS, "ho", rng)
    
    n_train_pairs = len(set(p["pair_id"] for p in train_packets))
    n_held_pairs = len(set(p["pair_id"] for p in held_packets))
    print(f"Generated: {len(train_packets)} train packets ({n_train_pairs} pairs), "
          f"{len(held_packets)} held-out packets ({n_held_pairs} pairs)", flush=True)
    
    # Save packets
    for name, pkts in [("train", train_packets), ("heldout", held_packets)]:
        with open(out_dir / f"{name}_packets.jsonl", "w") as f:
            for p in pkts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Load model
    from transformers import AutoTokenizer, DebertaV2Config
    from safetensors.torch import load_file
    model_path = pathlib.Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    
    def load_model():
        sys.path.insert(0, str(pathlib.Path("experiments/archive/functional_learning/scripts")))
        from context_credit_trainer import FrozenSlowPrivateDebertaV2ForMaskedLM
        cfg = DebertaV2Config.from_pretrained(str(model_path), local_files_only=True)
        cfg.private_adapter_bottleneck = args.private_bottleneck
        cfg.private_adapter_scale = args.private_scale
        cfg.private_adapter_enabled = True
        m = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
        sd = load_file(str(model_path / "model.safetensors"), device="cpu")
        m.load_state_dict(sd, strict=False)
        for mod in m.modules():
            if hasattr(mod, "private_adapter_enabled"):
                mod.private_adapter_enabled = True
        return m.to(device)
    
    # Import training utilities from the pilot
    sys.path.insert(0, str(pathlib.Path("experiments/archive/functional_learning/scripts")))
    from binding_training_pilot import (
        PairedPacketDataset, collate_fn, apply_focused_masking, evaluate_binding
    )
    from torch.utils.data import DataLoader
    
    # Create dataset and loader from TRAIN packets only
    ds = PairedPacketDataset(train_packets, tokenizer, seq_length=256)
    loader = DataLoader(ds, batch_size=min(16, len(ds)), shuffle=True, collate_fn=collate_fn)
    
    # Train with focused masking
    model = load_model()
    model.train()
    trainable = [p for n, p in model.named_parameters() if "private_adapter" in n]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)
    print(f"Training {len(trainable)} private adapter parameters on {n_train_pairs} train pairs", flush=True)
    
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    gen = torch.Generator(device=device)
    gen.manual_seed(43032)
    
    trajectory = []
    
    # Baseline evaluation on both train and held-out
    ev_train = evaluate_binding(model, tokenizer, train_packets, device)
    ev_held = evaluate_binding(model, tokenizer, held_packets, device)
    trajectory.append({
        "epoch": 0,
        "train": {"R_minus_N": ev_train["mean_retain_minus_neutral"],
                  "mean_update": ev_train["mean_update"],
                  "mean_retain": ev_train["mean_retain"],
                  "both_correct": ev_train["n_both_correct"],
                  "n_pairs": ev_train["n_pairs"]},
        "heldout": {"R_minus_N": ev_held["mean_retain_minus_neutral"],
                    "mean_update": ev_held["mean_update"],
                    "mean_retain": ev_held["mean_retain"],
                    "both_correct": ev_held["n_both_correct"],
                    "n_pairs": ev_held["n_pairs"]},
    })
    print(f"Epoch 0: TRAIN R-N={ev_train['mean_retain_minus_neutral']:+.2f} "
          f"both={ev_train['n_both_correct']}/{ev_train['n_pairs']} | "
          f"HELD R-N={ev_held['mean_retain_minus_neutral']:+.2f} "
          f"both={ev_held['n_both_correct']}/{ev_held['n_pairs']}", flush=True)
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_targets = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_pos = batch["answer_positions"]
            
            masked_inputs, labels = apply_focused_masking(
                input_ids, attention_mask, word_group, answer_pos,
                tokenizer.mask_token_id, 0.10, gen
            )
            n_targets = int((labels != -100).sum().item())
            if n_targets == 0:
                continue
            
            out = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += float(loss.detach()) * n_targets
            epoch_targets += n_targets
        
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ev_train = evaluate_binding(model, tokenizer, train_packets, device)
            ev_held = evaluate_binding(model, tokenizer, held_packets, device)
            entry = {
                "epoch": epoch,
                "loss": epoch_loss / max(1, epoch_targets),
                "train": {"R_minus_N": ev_train["mean_retain_minus_neutral"],
                          "mean_update": ev_train["mean_update"],
                          "mean_retain": ev_train["mean_retain"],
                          "both_correct": ev_train["n_both_correct"],
                          "n_pairs": ev_train["n_pairs"]},
                "heldout": {"R_minus_N": ev_held["mean_retain_minus_neutral"],
                            "mean_update": ev_held["mean_update"],
                            "mean_retain": ev_held["mean_retain"],
                            "both_correct": ev_held["n_both_correct"],
                            "n_pairs": ev_held["n_pairs"]},
            }
            trajectory.append(entry)
            print(f"Epoch {epoch}: loss={entry['loss']:.4f} | "
                  f"TRAIN R-N={ev_train['mean_retain_minus_neutral']:+.2f} "
                  f"both={ev_train['n_both_correct']}/{ev_train['n_pairs']} | "
                  f"HELD R-N={ev_held['mean_retain_minus_neutral']:+.2f} "
                  f"both={ev_held['n_both_correct']}/{ev_held['n_pairs']}", flush=True)
    
    # Save
    summary = {
        "status": "HELDOUT_GENERALIZATION",
        "n_train_pairs": n_train_pairs,
        "n_held_pairs": n_held_pairs,
        "n_train_packets": len(train_packets),
        "n_held_packets": len(held_packets),
        "epochs": args.epochs,
        "trajectory": trajectory,
        "baseline_train_R_minus_N": trajectory[0]["train"]["R_minus_N"],
        "baseline_held_R_minus_N": trajectory[0]["heldout"]["R_minus_N"],
        "final_train_R_minus_N": trajectory[-1]["train"]["R_minus_N"],
        "final_held_R_minus_N": trajectory[-1]["heldout"]["R_minus_N"],
        "final_train_both_correct": trajectory[-1]["train"]["both_correct"],
        "final_held_both_correct": trajectory[-1]["heldout"]["both_correct"],
    }
    
    with open(out_dir / "heldout_generalization.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(json.dumps({
        "status": summary["status"],
        "baseline_held_R_minus_N": summary["baseline_held_R_minus_N"],
        "final_held_R_minus_N": summary["final_held_R_minus_N"],
        "final_held_both_correct": f"{summary['final_held_both_correct']}/{n_held_pairs}",
        "final_train_both_correct": f"{summary['final_train_both_correct']}/{n_train_pairs}",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
