#!/usr/bin/env python3
"""research: Role-switch packet builder + CPU probe.

Generates balanced synthetic role-switch packets across 6 families where
the same alternatives exchange correctness under context changes. Includes:
  1. Balanced pair generation with train/held-out entity and template splits
  2. N-gram provenance check against official evaluation text
  3. WWM useful-density simulation with the legal40k tokenizer
  4. Tokenizer single-token analysis for target words

CPU-only. No pretrained model, official evaluation labels, or external tools.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, hashlib, itertools, os, pathlib, random, re, sys
from collections import Counter, defaultdict

USER_ROOT = pathlib.Path(".")
OUT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/role_switch_packets"
EVAL_ROOT = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data"
TOK_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"

SEED = 144042
N_PER_TEMPLATE = 20  # entity pairs per template

# ═══════════════════════════════════════════
# Vocabulary pools
# ═══════════════════════════════════════════

_NOUNS = sorted(["cat","dog","book","ball","cup","apple","chair","hat",
    "shoe","key","clock","spoon","plate","bottle","flower",
    "bird","fish","rock","leaf","pen","cake","bread",
    "car","bed","lamp","coin","ring","bell"])
NOUNS_TRAIN, NOUNS_HELDOUT = _NOUNS[:20], _NOUNS[20:]

CONTAINERS = sorted(["box","bag","basket","drawer","jar","bucket",
    "bowl","cabinet","chest","closet"])

PEOPLE_TRAIN = ["Tom","Sam","Ben","Max","Leo","Dan","Kim","Pat","Alex","Jen","Mia","Zoe"]
PEOPLE_HELDOUT = ["Eve","Lily","Jack","Kate","Anna","Luke"]

TRANSFER_THINGS = sorted(set(["ball","book","key","apple","hat","coin",
    "flower","pen","ring","cake","spoon","cup"]))
COUNTABLE = sorted(set(["apples","books","coins","flowers","balls","cups",
    "cards","shells","stones","stickers","marbles"]))

# ═══════════════════════════════════════════
# Templates — entity-swap families
# ═══════════════════════════════════════════
# ctx: {x}=primary role, {y}=secondary (swap x↔y for direction BA)
# con: {T}=correct target (=x if target_role==x, =y if target_role==y)
# con2: {O}=other entity (optional companion consequence)
# Extra vars: {P}=person, {thing}=transfer obj, {items}=countable, {c}=container

ENTITY_TEMPLATES = [
    # ── SPATIAL (pool=nouns, target_role=x: x is higher) ──
    {"family":"spatial","id":"sp01","style":"train","pool":"nouns","tr":"x",
     "ctx":"The {x} is on the shelf above the {y}.",
     "con":"The higher one is the {T}.","con2":"The lower one is the {O}."},
    {"family":"spatial","id":"sp02","style":"train","pool":"nouns","tr":"x",
     "ctx":"{P} placed the {x} above the {y} on the table.",
     "con":"The one in the higher spot is the {T}.","con2":None},
    {"family":"spatial","id":"sp03","style":"train","pool":"nouns","tr":"x",
     "ctx":"The {y} sits below the {x} on the rack.",
     "con":"The item on top is the {T}.","con2":"The item at the bottom is the {O}."},
    {"family":"spatial","id":"sp04","style":"train","pool":"nouns","tr":"x",
     "ctx":"The {y} was stacked under the {x}.",
     "con":"The one that ended up higher was the {T}.","con2":None},
    {"family":"spatial","id":"sp_h1","style":"held_out","pool":"nouns","tr":"x",
     "ctx":"The {x} rests over the {y} on the desk.",
     "con":"The upper one is the {T}.","con2":"The lower one is the {O}."},
    {"family":"spatial","id":"sp_h2","style":"held_out","pool":"nouns","tr":"x",
     "ctx":"The {y} is beneath the {x} in the stack.",
     "con":"The top one is the {T}.","con2":None},

    # ── TEMPORAL (pool=people, target_role=x: x is earlier) ──
    {"family":"temporal","id":"tm01","style":"train","pool":"people","tr":"x",
     "ctx":"{x} arrived at the park before {y}.",
     "con":"The one who came first was {T}.","con2":None},
    {"family":"temporal","id":"tm02","style":"train","pool":"people","tr":"x",
     "ctx":"{y} finished eating after {x} did.",
     "con":"The one who finished earlier was {T}.","con2":None},
    {"family":"temporal","id":"tm03","style":"train","pool":"people","tr":"x",
     "ctx":"{x} woke up earlier than {y} that morning.",
     "con":"The first one awake was {T}.","con2":None},
    {"family":"temporal","id":"tm04","style":"train","pool":"people","tr":"x",
     "ctx":"{x} left home before {y} did.",
     "con":"The one who left first was {T}.","con2":"The one still at home was {O}."},
    {"family":"temporal","id":"tm_h1","style":"held_out","pool":"people","tr":"x",
     "ctx":"{y} got to school later than {x}.",
     "con":"The earlier one was {T}.","con2":None},

    # ── TRANSFER (pool=people, target_role=y: y receives) ──
    {"family":"transfer","id":"tr01","style":"train","pool":"people","tr":"y",
     "ctx":"{x} gave the {thing} to {y}.",
     "con":"The one who had the {thing} afterward was {T}.","con2":"The one who no longer had it was {O}."},
    {"family":"transfer","id":"tr02","style":"train","pool":"people","tr":"y",
     "ctx":"{x} handed the {thing} to {y} at the table.",
     "con":"The one holding the {thing} now is {T}.","con2":None},
    {"family":"transfer","id":"tr03","style":"train","pool":"people","tr":"y",
     "ctx":"{x} passed the {thing} over to {y}.",
     "con":"The one who received it was {T}.","con2":None},
    {"family":"transfer","id":"tr_h1","style":"held_out","pool":"people","tr":"y",
     "ctx":"{x} lent the {thing} to {y} for the day.",
     "con":"The one borrowing the {thing} was {T}.","con2":None},

    # ── COMPARATIVE (pool=people, target_role=x: x has more) ──
    {"family":"comparative","id":"cm01","style":"train","pool":"people","tr":"x",
     "ctx":"{x} had more {items} than {y}.",
     "con":"The one with more was {T}.","con2":"The one with fewer was {O}."},
    {"family":"comparative","id":"cm02","style":"train","pool":"people","tr":"x",
     "ctx":"{y} collected fewer {items} than {x} did.",
     "con":"The one who collected more was {T}.","con2":None},
    {"family":"comparative","id":"cm03","style":"train","pool":"people","tr":"x",
     "ctx":"{x} found many more {items} than {y} did.",
     "con":"The one with the larger number was {T}.","con2":None},
    {"family":"comparative","id":"cm_h1","style":"held_out","pool":"people","tr":"x",
     "ctx":"{y} had not as many {items} as {x}.",
     "con":"The one with more {items} was {T}.","con2":None},

    # ── CONTAINER (HELD-OUT FAMILY — all templates held_out) ──
    {"family":"container","id":"cn01","style":"held_out","pool":"nouns","tr":"x",
     "ctx":"{P} put the {x} inside the {c} and left the {y} outside.",
     "con":"The thing inside the {c} is the {T}.","con2":"The thing outside is the {O}."},
    {"family":"container","id":"cn02","style":"held_out","pool":"nouns","tr":"x",
     "ctx":"The {x} is in the {c} but the {y} is not.",
     "con":"The one in the {c} is the {T}.","con2":None},
    {"family":"container","id":"cn03","style":"held_out","pool":"nouns","tr":"x",
     "ctx":"{P} placed the {x} into the {c} and the {y} on the table.",
     "con":"The thing that ended up in the {c} was the {T}.","con2":None},
]

# ── STATE-CHANGE (fixed alternatives, not entity swap) ──
STATE_TEMPLATES = [
    {"family":"state_change","id":"st01","style":"train",
     "ctx_0":"{P} opened the {c}.","ctx_1":"{P} closed the {c}.",
     "con":"After that, the {c} was {T}.","alt_0":"open","alt_1":"closed"},
    {"family":"state_change","id":"st02","style":"train",
     "ctx_0":"{P} turned on the lamp.","ctx_1":"{P} turned off the lamp.",
     "con":"The lamp was {T} after that.","alt_0":"on","alt_1":"off"},
    {"family":"state_change","id":"st03","style":"train",
     "ctx_0":"{P} filled the {c} with water.","ctx_1":"{P} emptied the {c}.",
     "con":"The {c} was {T}.","alt_0":"full","alt_1":"empty"},
    {"family":"state_change","id":"st_h1","style":"held_out",
     "ctx_0":"{P} locked the {c}.","ctx_1":"{P} unlocked the {c}.",
     "con":"The {c} was then {T}.","alt_0":"locked","alt_1":"unlocked"},
]

# ═══════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════

def _entity_pool(pool_name, style):
    if pool_name == "nouns":
        return NOUNS_TRAIN if style == "train" else NOUNS_TRAIN + NOUNS_HELDOUT
    else:
        return PEOPLE_TRAIN if style == "train" else PEOPLE_TRAIN + PEOPLE_HELDOUT

def _entity_split(a, b, pool_name):
    ho = set(NOUNS_HELDOUT if pool_name == "nouns" else PEOPLE_HELDOUT)
    return "held_out" if (a in ho or b in ho) else "train"

def _fmt(s, **kw):
    """Safe format: fill available keys, leave others."""
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s

def generate_entity_swap(rng):
    pairs, texts = [], []
    for tmpl in ENTITY_TEMPLATES:
        pool = _entity_pool(tmpl["pool"], tmpl["style"])
        combos = list(itertools.combinations(pool, 2))
        rng.shuffle(combos)
        for a, b in combos[:N_PER_TEMPLATE]:
            P = rng.choice(PEOPLE_TRAIN)
            thing = rng.choice(TRANSFER_THINGS)
            items = rng.choice(COUNTABLE)
            c = rng.choice(CONTAINERS)
            kw = dict(P=P, thing=thing, items=items, c=c)

            ctx_AB = _fmt(tmpl["ctx"], x=a, y=b, **kw)
            ctx_BA = _fmt(tmpl["ctx"], x=b, y=a, **kw)

            if tmpl["tr"] == "x":
                cor_AB, cor_BA, oth_AB, oth_BA = a, b, b, a
            else:
                cor_AB, cor_BA, oth_AB, oth_BA = b, a, a, b

            con_AB = _fmt(tmpl["con"], T=cor_AB, O=oth_AB, **kw)
            con_BA = _fmt(tmpl["con"], T=cor_BA, O=oth_BA, **kw)
            full_AB = ctx_AB + " " + con_AB
            full_BA = ctx_BA + " " + con_BA
            if tmpl.get("con2"):
                full_AB += " " + _fmt(tmpl["con2"], T=cor_AB, O=oth_AB, **kw)
                full_BA += " " + _fmt(tmpl["con2"], T=cor_BA, O=oth_BA, **kw)

            con_masked = _fmt(tmpl["con"], T="__TARGET__", O="__OTHER__", **kw)

            pair = {
                "family": tmpl["family"], "template_id": tmpl["id"],
                "style": tmpl["style"],
                "entity_a": a, "entity_b": b, "person": P,
                "extras": kw,
                "alt_0": a, "alt_1": b,
                "correct_AB": cor_AB, "correct_BA": cor_BA,
                "context_AB": ctx_AB, "context_BA": ctx_BA,
                "consequence_masked": con_masked,
                "full_text_AB": full_AB, "full_text_BA": full_BA,
                "entity_split": _entity_split(a, b, tmpl["pool"]),
                "alt_type": "entity",
            }
            pairs.append(pair)
            for d, ft in [("AB", full_AB), ("BA", full_BA)]:
                texts.append({"text": ft, "family": tmpl["family"],
                    "template_id": tmpl["id"], "style": tmpl["style"],
                    "direction": d, "word_count": len(ft.split())})
    return pairs, texts

def generate_state_change(rng):
    pairs, texts = [], []
    for tmpl in STATE_TEMPLATES:
        combos = list(itertools.product(PEOPLE_TRAIN, CONTAINERS))
        rng.shuffle(combos)
        for P, c in combos[:N_PER_TEMPLATE]:
            kw = dict(P=P, c=c)
            ctx_0 = _fmt(tmpl["ctx_0"], **kw)
            ctx_1 = _fmt(tmpl["ctx_1"], **kw)
            a0, a1 = tmpl["alt_0"], tmpl["alt_1"]
            con_0 = _fmt(tmpl["con"], T=a0, **kw)
            con_1 = _fmt(tmpl["con"], T=a1, **kw)
            full_0 = ctx_0 + " " + con_0
            full_1 = ctx_1 + " " + con_1
            con_masked = _fmt(tmpl["con"], T="__TARGET__", **kw)

            pair = {
                "family": tmpl["family"], "template_id": tmpl["id"],
                "style": tmpl["style"],
                "entity_a": f"{P}_{c}", "entity_b": f"{P}_{c}",
                "person": P,
                "extras": kw,
                "alt_0": a0, "alt_1": a1,
                "correct_AB": a0, "correct_BA": a1,
                "context_AB": ctx_0, "context_BA": ctx_1,
                "consequence_masked": con_masked,
                "full_text_AB": full_0, "full_text_BA": full_1,
                "entity_split": "train",
                "alt_type": "state",
            }
            pairs.append(pair)
            for d, ft in [("AB", full_0), ("BA", full_1)]:
                texts.append({"text": ft, "family": tmpl["family"],
                    "template_id": tmpl["id"], "style": tmpl["style"],
                    "direction": d, "word_count": len(ft.split())})
    return pairs, texts

# ═══════════════════════════════════════════
# Provenance check
# ═══════════════════════════════════════════

def extract_ngrams(text, n):
    words = text.lower().split()
    return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

def load_eval_text(root):
    """Collect all text content from official evaluation data."""
    all_text = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if fn.endswith(('.jsonl','.json','.txt','.csv','.tsv')):
                try:
                    with open(fp, encoding='utf-8', errors='ignore') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            # Try JSONL
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    for v in obj.values():
                                        if isinstance(v, str) and len(v) > 10:
                                            all_text.append(v)
                                elif isinstance(obj, list):
                                    for item in obj:
                                        if isinstance(item, str) and len(item) > 10:
                                            all_text.append(item)
                            except json.JSONDecodeError:
                                if len(line) > 10:
                                    all_text.append(line)
                except Exception:
                    pass
    return all_text

def provenance_check(gen_texts, eval_root):
    print("Loading official evaluation text...", flush=True)
    eval_strings = load_eval_text(eval_root)
    print(f"  Loaded {len(eval_strings)} eval text segments", flush=True)

    results = {}
    for n in [7, 8, 10]:
        gen_ng = set()
        for t in gen_texts:
            gen_ng.update(extract_ngrams(t["text"], n))
        eval_ng = set()
        for s in eval_strings:
            eval_ng.update(extract_ngrams(s, n))
        overlap = gen_ng & eval_ng
        results[f"{n}gram"] = {
            "gen_count": len(gen_ng), "eval_count": len(eval_ng),
            "overlap_count": len(overlap),
            "overlap_samples": [" ".join(ng) for ng in sorted(overlap)[:20]],
        }
        print(f"  {n}-gram: gen={len(gen_ng)}, eval={len(eval_ng)}, overlap={len(overlap)}", flush=True)
    return results

# ═══════════════════════════════════════════
# WWM density simulation
# ═══════════════════════════════════════════

def wwm_density_simulation(pairs, tok_dir, n_passes=10, mask_prob=0.15):
    """Simulate WWM masking and count useful target events."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    except Exception as e:
        return {"error": str(e)}

    rng = random.Random(144099)
    stats = defaultdict(lambda: {"total_words": 0, "total_tokens": 0,
        "useful_masked_events": 0, "total_masked_groups": 0,
        "useful_density": 0.0, "pairs": 0})

    # Check single-token encoding of all target alternatives
    all_alts = set()
    for p in pairs:
        all_alts.add(p["alt_0"])
        all_alts.add(p["alt_1"])
    token_analysis = {}
    for w in sorted(all_alts):
        ids = tokenizer.encode(w, add_special_tokens=False)
        token_analysis[w] = {"token_ids": ids, "n_tokens": len(ids),
            "single_token": len(ids) == 1}

    for p in pairs:
        family = p["family"]
        stats[family]["pairs"] += 1

        for direction in ["AB", "BA"]:
            full = p[f"full_text_{direction}"]
            words = full.split()
            n_words = len(words)
            stats[family]["total_words"] += n_words

            # Tokenize to get word groups
            token_ids = []
            word_groups = []  # maps each token to its word index
            for wi, w in enumerate(words):
                toks = tokenizer.encode(w, add_special_tokens=False)
                for t in toks:
                    token_ids.append(t)
                    word_groups.append(wi)
            n_tokens = len(token_ids)
            stats[family]["total_tokens"] += n_tokens

            # Identify useful target words (the correct alternative in consequence)
            correct = p[f"correct_{direction}"]
            other_alt = p["alt_1"] if correct == p["alt_0"] else p["alt_0"]
            # Find word indices that contain the target in the consequence
            # The consequence starts after the context
            ctx = p[f"context_{direction}"]
            ctx_words = len(ctx.split())
            useful_word_indices = set()
            for wi in range(ctx_words, n_words):
                w_lower = words[wi].lower().rstrip(".,!?;:")
                if w_lower == correct.lower() or w_lower == other_alt.lower():
                    useful_word_indices.add(wi)

            # Simulate n_passes of WWM
            unique_word_groups_set = sorted(set(word_groups))
            for _ in range(n_passes):
                masked_groups = set()
                for wg in unique_word_groups_set:
                    if rng.random() < mask_prob:
                        masked_groups.add(wg)
                stats[family]["total_masked_groups"] += len(masked_groups)
                useful_hit = len(masked_groups & useful_word_indices)
                stats[family]["useful_masked_events"] += useful_hit

    # Compute densities
    for fam, s in stats.items():
        total_word_exposures = s["total_words"]  # across n_passes directions
        if total_word_exposures > 0:
            s["useful_density"] = s["useful_masked_events"] / (total_word_exposures * n_passes)
        s["useful_density_per_pass"] = (s["useful_masked_events"] / n_passes
            if s["pairs"] > 0 else 0)

    return {"per_family": dict(stats), "token_analysis": token_analysis,
            "n_passes": n_passes, "mask_prob": mask_prob}

# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Generate
    print("Generating entity-swap pairs...", flush=True)
    e_pairs, e_texts = generate_entity_swap(rng)
    print(f"  Entity-swap: {len(e_pairs)} pairs, {len(e_texts)} texts", flush=True)

    print("Generating state-change pairs...", flush=True)
    s_pairs, s_texts = generate_state_change(rng)
    print(f"  State-change: {len(s_pairs)} pairs, {len(s_texts)} texts", flush=True)

    all_pairs = e_pairs + s_pairs
    all_texts = e_texts + s_texts

    # Balance summary
    fam_counts = defaultdict(lambda: {"train": 0, "held_out": 0})
    for p in all_pairs:
        fam_counts[p["family"]][p["style"]] += 1
    dir_counts = Counter(t["direction"] for t in all_texts)
    total_words = sum(t["word_count"] for t in all_texts)
    train_words = sum(t["word_count"] for t in all_texts if t["style"] == "train")

    print(f"\nTotal: {len(all_pairs)} pairs, {len(all_texts)} texts, {total_words} words", flush=True)
    print(f"Training texts: {sum(1 for t in all_texts if t['style']=='train')} ({train_words} words)", flush=True)
    print(f"Direction balance: {dict(dir_counts)}", flush=True)
    for fam, c in sorted(fam_counts.items()):
        print(f"  {fam}: train={c['train']}, held_out={c['held_out']}", flush=True)

    # Entity split summary
    esplit = Counter(p["entity_split"] for p in all_pairs)
    print(f"Entity split: {dict(esplit)}", flush=True)

    # Save texts JSONL
    texts_path = OUT_DIR / "training_texts.jsonl"
    with open(texts_path, "w") as f:
        for t in all_texts:
            f.write(json.dumps(t) + "\n")

    # Save scoring pairs JSONL
    pairs_path = OUT_DIR / "scoring_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for i, p in enumerate(all_pairs):
            p_out = {**p, "pair_id": i}
            f.write(json.dumps(p_out) + "\n")

    # Provenance check
    print("\n── Provenance check ──", flush=True)
    prov = provenance_check(all_texts, EVAL_ROOT)

    # WWM density simulation
    print("\n── WWM density simulation ──", flush=True)
    wwm = wwm_density_simulation(all_pairs, TOK_DIR)
    if "error" not in wwm:
        for fam, s in sorted(wwm["per_family"].items()):
            print(f"  {fam}: pairs={s['pairs']}, total_words={s['total_words']}, "
                  f"useful_masked/pass={s['useful_density_per_pass']:.1f}, "
                  f"useful_density={s['useful_density']:.4f}", flush=True)
        # Token analysis
        multi = [w for w, a in wwm["token_analysis"].items() if not a["single_token"]]
        print(f"  Multi-token targets: {multi if multi else 'none'}", flush=True)
    else:
        print(f"  WWM error: {wwm['error']}", flush=True)

    # Save manifest
    manifest = {
        "status": "ROLE_SWITCH_PACKETS",
        "seed": SEED, "n_per_template": N_PER_TEMPLATE,
        "total_pairs": len(all_pairs), "total_texts": len(all_texts),
        "total_words": total_words, "train_words": train_words,
        "family_counts": {k: dict(v) for k, v in fam_counts.items()},
        "direction_balance": dict(dir_counts),
        "entity_split": dict(esplit),
        "provenance": prov,
        "wwm_density": wwm if "error" not in wwm else {"error": wwm["error"]},
        "vocabulary": {
            "nouns_train": NOUNS_TRAIN, "nouns_heldout": NOUNS_HELDOUT,
            "people_train": PEOPLE_TRAIN, "people_heldout": PEOPLE_HELDOUT,
            "containers": CONTAINERS,
        },
        "code_hash": hashlib.sha256(
            _public_path('experiments/archive/representation_and_objectives/scripts/role_switch_packet_builder.py').read_bytes()).hexdigest(),
        "output_files": {
            "training_texts": str(texts_path),
            "scoring_pairs": str(pairs_path),
        },
    }
    manifest_path = OUT_DIR / "packet_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved: {texts_path}, {pairs_path}, {manifest_path}", flush=True)
    print(json.dumps({"event": "packet_builder_done",
        "pairs": len(all_pairs), "texts": len(all_texts),
        "words": total_words, "train_words": train_words,
        "manifest": str(manifest_path)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
