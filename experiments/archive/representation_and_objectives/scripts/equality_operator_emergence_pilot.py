#!/usr/bin/env python3
"""research: equality-operator emergence pilot.

This CPU pilot separates three questions that research/298 conflated:

1. Does a matcher have an equality *prior* before any finite evidence?
2. Can finite name-level equality evidence train a position-sensitive equality
   operator that generalizes to unseen names in raw events?
3. What lower-level coverage is required: are held names with characters absent
   from the equality evidence fundamentally outside the learned coordinate?

The pilot deliberately does not run relational gauge training. It tests only the
upstream identity route: hard assignment of candidate/other names among all raw
text tokens, including non-name lexical distractors. This is the lowest-cost
experiment that can determine whether another relational GPU run would be
scientifically interpretable.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, json, math, random, sys, time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the existing harness without relying on package layout.
SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/training/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import raw_span_discovery_probe as base  # noqa: E402

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_OUT = PROJECT / "data/equality_operator_emergence_pilot"
MAX_LEN = 10
N_CHARS = 28


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def name_chars(name: str, max_len: int = MAX_LEN) -> torch.Tensor:
    ids = base.name_to_char_ids(name)[:max_len]
    ids = ids + [0] * max(0, max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def forms_to_tensor(forms: Sequence[str], device: torch.device) -> torch.Tensor:
    out = torch.zeros((len(forms), MAX_LEN), dtype=torch.long, device=device)
    for i, f in enumerate(forms):
        if f:
            out[i] = name_chars(f).to(device)
    return out


def special_mask(forms: Sequence[str], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(len(forms), dtype=torch.bool, device=device)
    for i, f in enumerate(forms):
        # Preserve the research convention: empty and explicit boundary/hypothesis
        # forms are not valid name assignments, but ordinary lexical words remain
        # competitors. This keeps entity detection hard.
        if not f or f in {"<hyp>", "<s>", "</s>"}:
            mask[i] = True
    if len(forms):
        mask[0] = True
        mask[-1] = True
    return mask


class SharedPosMatcher(nn.Module):
    """Position-sensitive equality with a built-in same-symbol prior.

    Token-side and query-side character embeddings are the same table. Random
    initialization already makes a character most similar to itself because the
    self dot product is large while cross dot products are centered near zero.
    """
    def __init__(self, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(N_CHARS, dim, padding_idx=0)
        self.length_penalty = nn.Parameter(torch.tensor(-3.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def scores(self, tok_chars: torch.Tensor, qry_chars: torch.Tensor) -> torch.Tensor:
        tok_emb = self.emb(tok_chars)
        qry_emb = self.emb(qry_chars)
        return pos_scores_from_emb(tok_chars, qry_chars, tok_emb, qry_emb,
                                   self.length_penalty, self.temperature)


class DualPosMatcher(nn.Module):
    """Position-sensitive equality without shared token/query embeddings.

    This removes the zero-shot equality prior while preserving the right
    architectural decomposition (same positions compared, soft AND over all
    positions). If finite equality evidence is sufficient, training should align
    the two character coordinates for observed characters.
    """
    def __init__(self, dim: int = 16):
        super().__init__()
        self.tok_emb = nn.Embedding(N_CHARS, dim, padding_idx=0)
        self.qry_emb = nn.Embedding(N_CHARS, dim, padding_idx=0)
        self.length_penalty = nn.Parameter(torch.tensor(-3.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def scores(self, tok_chars: torch.Tensor, qry_chars: torch.Tensor) -> torch.Tensor:
        tok_emb = self.tok_emb(tok_chars)
        qry_emb = self.qry_emb(qry_chars)
        return pos_scores_from_emb(tok_chars, qry_chars, tok_emb, qry_emb,
                                   self.length_penalty, self.temperature)


def pos_scores_from_emb(tok_chars: torch.Tensor, qry_chars: torch.Tensor,
                        tok_emb: torch.Tensor, qry_emb: torch.Tensor,
                        length_penalty: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """research-style position-product scores for all event tokens."""
    sim = (tok_emb * qry_emb.unsqueeze(0)).sum(dim=-1)
    tok_mask = tok_chars > 0
    qry_mask = qry_chars > 0
    both_valid = tok_mask & qry_mask.unsqueeze(0)
    tok_len = tok_mask.sum(dim=1)
    qry_len = qry_mask.sum()
    same_length = (tok_len == qry_len).float()
    match_prob = torch.sigmoid(sim / temperature.abs().clamp(min=0.1))
    valid_match = match_prob * both_valid.float() + (1.0 - both_valid.float())
    product = valid_match.prod(dim=1)
    len_factor = same_length + (1 - same_length) * torch.sigmoid(length_penalty)
    return product * len_factor


def unique_events(states, comps) -> List[Tuple[str, Tuple[str, str]]]:
    seen = set(); out = []
    for q in states:
        if getattr(q, "is_changed", False):
            k = (q.event, tuple(q.names))
            if k not in seen:
                seen.add(k); out.append(k)
    for c in comps:
        for ev in [c.event1, c.event2]:
            k = (ev, tuple(c.names))
            if k not in seen:
                seen.add(k); out.append(k)
    return out


def lower_names(examples: Sequence[Tuple[str, Tuple[str, str]]]) -> List[str]:
    names = []
    for _, ns in examples:
        names.extend([n.lower() for n in ns])
    return sorted(set(names))


def letters_in_names(names: Iterable[str]) -> List[str]:
    return sorted(set(ch for n in names for ch in n.lower() if "a" <= ch <= "z"))


def chars_to_ids(chars: Iterable[str]) -> List[int]:
    ids = []
    for ch in chars:
        if "a" <= ch <= "z":
            ids.append(ord(ch) - ord("a") + 1)
    return sorted(set(ids))


def synthetic_examples(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, Tuple[str, str]]]:
    out = []
    for a, b in pairs:
        ev = f"<s> {a} carried the dax near {b} before {b} handed the peb to {a} </s>"
        out.append((ev, (a, b)))
    return out


def eval_assignment(matcher: nn.Module, examples: Sequence[Tuple[str, Tuple[str, str]]],
                    device: torch.device) -> Dict[str, float]:
    matcher.eval()
    both_ok = rel_ok = det_ok = 0
    total_queries = 0
    name_minus_nonname: List[float] = []
    true_scores: List[float] = []
    max_nonname_scores: List[float] = []
    wrong_argmax_nonname = 0
    with torch.no_grad():
        for text, names in examples:
            _, forms = base.raw_tokenize(text)
            forms_l = [str(f).lower() for f in forms]
            tcp = forms_to_tensor(forms_l, device)
            smask = special_mask(forms_l, device)
            positions = {n.lower(): [i for i, f in enumerate(forms_l) if f == n.lower()] for n in names}
            valid_name_positions = set(i for ps in positions.values() for i in ps)
            nonname_positions = [i for i in range(len(forms_l)) if i not in valid_name_positions and not bool(smask[i].item())]
            for cand in names:
                other = names[1] if cand == names[0] else names[0]
                cand_l = cand.lower(); other_l = other.lower()
                if not positions.get(cand_l) or not positions.get(other_l):
                    continue
                cand_pos = positions[cand_l][0]
                other_pos = positions[other_l][0]
                q = name_chars(cand_l).to(device)
                s = matcher.scores(tcp, q).clone()
                s[smask] = -1.0
                arg = int(torch.argmax(s).item())
                total_queries += 1
                both_ok += int(arg == cand_pos)
                rel_ok += int(float(s[cand_pos]) > float(s[other_pos]))
                det_ok += int(arg in valid_name_positions)
                wrong_argmax_nonname += int(arg in nonname_positions)
                if nonname_positions:
                    mn = float(torch.max(s[nonname_positions]).cpu())
                    ts = float(s[cand_pos].cpu())
                    name_minus_nonname.append(ts - mn)
                    true_scores.append(ts); max_nonname_scores.append(mn)
    def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
    return {
        "n_queries": total_queries,
        "assign_acc": both_ok / total_queries if total_queries else float("nan"),
        "relative_name_acc": rel_ok / total_queries if total_queries else float("nan"),
        "argmax_is_a_name_frac": det_ok / total_queries if total_queries else float("nan"),
        "argmax_nonname_frac": wrong_argmax_nonname / total_queries if total_queries else float("nan"),
        "mean_true_minus_max_nonname": mean(name_minus_nonname),
        "mean_true_score": mean(true_scores),
        "mean_max_nonname_score": mean(max_nonname_scores),
    }


def train_name_level(matcher: nn.Module, train_examples: Sequence[Tuple[str, Tuple[str, str]]],
                     device: torch.device, epochs: int, lr: float, print_every: int = 0) -> List[Dict[str, float]]:
    matcher.to(device)
    opt = torch.optim.AdamW(matcher.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(29901)
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        matcher.train()
        exs = list(train_examples); rng.shuffle(exs)
        losses = []
        opt.zero_grad(set_to_none=True)
        for text, names in exs:
            _, forms = base.raw_tokenize(text)
            forms_l = [str(f).lower() for f in forms]
            tcp = forms_to_tensor(forms_l, device)
            smask = special_mask(forms_l, device)
            for cand in names:
                cand_l = cand.lower()
                q = name_chars(cand_l).to(device)
                s = matcher.scores(tcp, q).clamp(1e-7, 1-1e-7)
                target = torch.zeros(len(forms_l), dtype=torch.float32, device=device)
                for i, f in enumerate(forms_l):
                    if f == cand_l:
                        target[i] = 1.0
                target[smask] = 0.0
                losses.append(F.binary_cross_entropy(s, target, reduction="mean"))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(matcher.parameters(), 10.0)
        opt.step()
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            rec = {"epoch": ep, "loss": float(loss.detach().cpu()), "elapsed": round(time.time() - t0, 2)}
            hist.append(rec)
            if print_every:
                print(json.dumps({"name_train": rec}), flush=True)
    return hist


def train_char_pairs(matcher: DualPosMatcher, char_ids: Sequence[int], device: torch.device,
                     epochs: int, lr: float, print_every: int = 0) -> List[Dict[str, float]]:
    matcher.to(device)
    opt = torch.optim.AdamW(matcher.parameters(), lr=lr, weight_decay=0.0)
    pairs = [(a, b, float(a == b)) for a in char_ids for b in char_ids]
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        random.shuffle(pairs)
        losses = []
        opt.zero_grad(set_to_none=True)
        for a, b, y in pairs:
            tok = torch.zeros((1, MAX_LEN), dtype=torch.long, device=device)
            qry = torch.zeros(MAX_LEN, dtype=torch.long, device=device)
            tok[0, 0] = a; qry[0] = b
            s = matcher.scores(tok, qry).clamp(1e-7, 1-1e-7)[0]
            target = torch.tensor(y, dtype=torch.float32, device=device)
            losses.append(F.binary_cross_entropy(s, target, reduction="mean"))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(matcher.parameters(), 10.0)
        opt.step()
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            with torch.no_grad():
                acc = 0
                for a, b, y in pairs:
                    tok = torch.zeros((1, MAX_LEN), dtype=torch.long, device=device)
                    qry = torch.zeros(MAX_LEN, dtype=torch.long, device=device)
                    tok[0, 0] = a; qry[0] = b
                    pred = float(matcher.scores(tok, qry)[0].detach().cpu()) >= 0.5
                    acc += int(pred == bool(y))
            rec = {"epoch": ep, "loss": float(loss.detach().cpu()),
                   "pair_acc": acc / len(pairs), "elapsed": round(time.time() - t0, 2)}
            hist.append(rec)
            if print_every:
                print(json.dumps({"char_pair_train": rec}), flush=True)
    return hist


def evaluate_model(label: str, matcher: nn.Module, eval_sets: Dict[str, List[Tuple[str, Tuple[str, str]]]],
                   device: torch.device) -> Dict[str, Dict[str, float]]:
    return {name: eval_assignment(matcher, exs, device) for name, exs in eval_sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    ap.add_argument("--arm", default=base.DEFAULT_ARM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--name-epochs", type=int, default=80)
    ap.add_argument("--char-epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=29900)
    ap.add_argument("--print-every", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cpu")
    ts, tc, es, ec, _, counts = base.load_dataset(args.data_root, args.arm)
    train_events = unique_events(ts, tc)
    eval_events = unique_events(es, ec)
    train_names = lower_names(train_events)
    eval_names = lower_names(eval_events)
    train_letters = letters_in_names(train_names)
    eval_letters = letters_in_names(eval_names)
    missing_eval_letters = sorted(set(eval_letters) - set(train_letters))

    fresh_known = synthetic_examples([
        ("Lira", "Noma"), ("Tera", "Seli"), ("Mavi", "Rona"), ("Nilo", "Kera"),
        ("Tomi", "Pava"), ("Risa", "Leva"), ("Fero", "Jina"), ("Soma", "Veli"),
    ])
    fresh_unseen = synthetic_examples([
        ("Zuby", "Quin"), ("Hugo", "Celia"), ("Yara", "Bex"), ("Gwen", "Duca"),
        ("Zara", "Hugo"), ("Caleb", "June"), ("Byron", "Cora"), ("Quora", "Zed"),
    ])
    near_known = synthetic_examples([
        ("Milo", "Mila"), ("Sara", "Sora"), ("Lena", "Lina"), ("Rina", "Risa"),
        ("Tomas", "Tomar"), ("Keira", "Keria"), ("Noel", "Niel"), ("Pavel", "Pavel"),
    ])
    # Avoid identical pair in near-known; replace if tokenization sees both same.
    near_known[-1] = synthetic_examples([("Pavel", "Pavin")])[0]

    eval_sets = {
        "train_events": train_events,
        "official_held_events": eval_events,
        "fresh_known_chars": fresh_known,
        "fresh_unseen_chars": fresh_unseen,
        "near_known_names": near_known,
    }

    results = {}
    histories = {}

    # 1. Shared embedding prior: position structure plus same-table character identity.
    torch.manual_seed(args.seed)
    shared = SharedPosMatcher(args.dim).to(device)
    results["shared_pos_initial"] = evaluate_model("shared_pos_initial", shared, eval_sets, device)

    # 2. Dual embedding with no prior, before training.
    torch.manual_seed(args.seed)
    dual0 = DualPosMatcher(args.dim).to(device)
    results["dual_pos_initial"] = evaluate_model("dual_pos_initial", dual0, eval_sets, device)

    # 3. Dual embedding trained only from name-in-event equality on train names.
    dual_name = copy.deepcopy(dual0)
    histories["dual_name_level"] = train_name_level(dual_name, train_events, device, args.name_epochs, args.lr, args.print_every)
    results["dual_pos_name_level_train"] = evaluate_model("dual_pos_name_level_train", dual_name, eval_sets, device)

    # 4. Dual embedding with character-pair anchors only for letters appearing in train names.
    dual_char_train = copy.deepcopy(dual0)
    histories["dual_train_alphabet_char_pairs"] = train_char_pairs(
        dual_char_train, chars_to_ids(train_letters), device, args.char_epochs, args.lr, args.print_every)
    results["dual_pos_train_alphabet_char_pairs"] = evaluate_model("dual_pos_train_alphabet_char_pairs", dual_char_train, eval_sets, device)

    # 5. Dual embedding with character-pair anchors for all a-z letters.
    dual_char_full = copy.deepcopy(dual0)
    histories["dual_full_alphabet_char_pairs"] = train_char_pairs(
        dual_char_full, list(range(1, 27)), device, args.char_epochs, args.lr, args.print_every)
    results["dual_pos_full_alphabet_char_pairs"] = evaluate_model("dual_pos_full_alphabet_char_pairs", dual_char_full, eval_sets, device)

    meta = {
        "counts": counts,
        "n_train_events": len(train_events),
        "n_eval_events": len(eval_events),
        "train_names": train_names,
        "official_eval_names": eval_names,
        "train_letters": train_letters,
        "official_eval_letters": eval_letters,
        "official_eval_letters_absent_from_train_names": missing_eval_letters,
        "dim": args.dim,
        "name_epochs": args.name_epochs,
        "char_epochs": args.char_epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    out = args.out
    write_json(out / "equality_operator_emergence.json", {"meta": meta, "histories": histories, "results": results})

    lines = [
        "# research equality-operator emergence pilot", "",
        "This CPU pilot tests the identity route upstream of relational gauge transport.", "",
        "## Character coverage", "",
        f"Train-name letters: `{''.join(train_letters)}`", "",
        f"Official held-name letters: `{''.join(eval_letters)}`", "",
        f"Held letters absent from train names: `{''.join(missing_eval_letters)}`", "",
        "## Assignment accuracy", "",
        "Hard assignment is correct only when the candidate query selects the true candidate token among all raw tokens, not just over the other name.", "",
        "| model | train events | official held | fresh known chars | fresh unseen chars | near known names | held rel-name | held true-maxnonname |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, res in results.items():
        lines.append(
            f"| {model_name} "
            f"| {res['train_events']['assign_acc']:.3f} "
            f"| {res['official_held_events']['assign_acc']:.3f} "
            f"| {res['fresh_known_chars']['assign_acc']:.3f} "
            f"| {res['fresh_unseen_chars']['assign_acc']:.3f} "
            f"| {res['near_known_names']['assign_acc']:.3f} "
            f"| {res['official_held_events']['relative_name_acc']:.3f} "
            f"| {res['official_held_events']['mean_true_minus_max_nonname']:.4f} |"
        )
    lines += ["", "## Interpretation hooks", "",
              "- `shared_pos_initial` measures the built-in same-symbol prior supplied by shared character embeddings plus position-wise product.",
              "- `dual_pos_initial` removes that prior while preserving position-sensitive structure.",
              "- `dual_pos_name_level_train` asks whether finite name-level equality evidence from train events can align token/query character coordinates.",
              "- `dual_pos_train_alphabet_char_pairs` asks whether lower-level equality anchors for only observed train letters are enough.",
              "- `dual_pos_full_alphabet_char_pairs` asks whether complete alphabet-level finite equality evidence restores broad held-name assignment.", ""]
    (out / "equality_operator_emergence.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "EQUALITY_OPERATOR_EMERGENCE_COMPLETE",
                      "md": str(out / "equality_operator_emergence.md"),
                      "json": str(out / "equality_operator_emergence.json")}, indent=2))


if __name__ == "__main__":
    main()
