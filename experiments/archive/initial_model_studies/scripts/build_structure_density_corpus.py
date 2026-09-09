#!/usr/bin/env python3
"""Build structure-density selected arms from the official BabyLM corpus.

Design principles:
- All arms match unique corpus amount, source-file proportions, repetition
  distribution, and training geometry.
- The only variable between arms is WHICH windows within each source are
  selected, ranked by structure-density score.
- Before training, report per-arm structure densities to confirm separation.

Arms:
  high_entity_state   : top quartile by (entity + pronoun + state/transfer) within each source
  high_physical       : top quartile by (physical + spatial + action) within each source
  matched_low         : bottom quartile by composite score within each source
  uniform             : random quartile from each source (reference)

All arms:
- same unique word budget from each source (= fraction × source_words)
- same number of repetition epochs to reach target exposure
- same training steps, batch, optimizer, seeds
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

# ──── Lexicon sets ────

STOP = {
    'a','an','the','and','or','but','if','then','else','when','while','of','to','in','on','at','by','for','from','with','without','as','is','are','was','were','be','been','being','am','do','does','did','done','have','has','had','having','can','could','will','would','shall','should','may','might','must','not','no','yes','than','that','this','these','those','it','its','he','she','they','them','his','her','their','we','us','you','your','i','me','my','mine','our','ours','who','whom','which','what','where','why','how','there','here','also','more','most','other','some','any','all','one','two','first','second','new','old','many','much','such','into','over','under','after','before','about','up','down','out','off','only','through','between','during','against','within','because'
}
PRONOUNS = {'he','she','it','they','him','her','them','his','hers','its','their','theirs','himself','herself','itself','themselves','who','whom','whose','which','that','this','these','those','i','me','my','mine','we','us','our','ours','you','your','yours'}
STATE_TRANSFER = {'put','puts','placed','place','places','moved','move','moves','moving','transfer','transferred','give','gave','given','take','took','taken','bring','brought','send','sent','carry','carried','drop','dropped','fall','fell','fallen','break','broke','broken','open','opened','close','closed','change','changed','become','became','turn','turned','keep','kept','hold','held','store','stored','remove','removed','replace','replaced','contain','contains','contained','enter','entered','leave','left','push','pushed','pull','pulled','throw','threw','thrown','pick','picked','catch','caught','grab','grabbed','lift','lifted','pour','poured','fill','filled','empty','emptied','add','added','lose','lost','find','found','hide','hid','hidden','set','sit','sat','stand','stood','lay','laid','hang','hung','wear','wore','worn'}
SPATIAL = {'in','on','at','under','over','above','below','inside','outside','near','between','behind','front','beside','around','through','across','within','into','onto','from','to','left','right','north','south','east','west','up','down','there','here','top','bottom','middle','next','far','close','away','back','forward','toward','towards'}
PHYSICAL = {'glass','wood','wooden','metal','plastic','stone','rock','water','ice','fire','air','paper','cloth','rubber','fragile','heavy','light','hot','cold','warm','wet','dry','hard','soft','solid','liquid','gas','round','flat','sharp','smooth','rough','break','burn','melt','freeze','float','sink','fall','rise','push','pull','roll','slide','throw','hit','cut','bend','stretch','squeeze','bounce','crack','spill','pour','mix','stick','tear'}
SOCIAL = {'said','say','says','told','tell','asked','ask','asks','answer','answered','reply','replied','talk','spoke','speak','friend','mother','father','child','boy','girl','man','woman','people','person','team','group','family','king','queen','want','wanted','think','thought','know','knew','feel','felt','like','love','hate','need','help','helped','thank','please','sorry','happy','sad','angry','afraid','scared','surprised','worried'}
CAUSAL_TEMPORAL = {'because','cause','caused','therefore','so','if','then','after','before','during','while','when','until','since','although','though','however','result','results','finally','suddenly','immediately','soon','later','earlier','already','still','yet','now','again','once','first','next','last','begin','began','begun','start','started','end','ended','finish','finished','happen','happened','make','made','let'}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
CAP_RE = re.compile(r"^[A-Z][a-z]+")


@dataclass
class Window:
    source_file: str
    start_line: int
    end_line: int
    text: str
    words: int
    # scores filled after construction
    entity_state_score: float = 0.0
    physical_score: float = 0.0
    social_causal_score: float = 0.0
    composite_score: float = 0.0


def tokenize_lower(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


def score_window(w: Window) -> None:
    """Score a window on multiple structure axes, normalized per word."""
    toks = WORD_RE.findall(w.text)
    n = max(1, len(toks))
    low = [t.lower() for t in toks]

    # Entity cues: repeated capitalized tokens (non-initial position) + pronouns + state verbs
    cap_counts: dict[str, int] = collections.Counter()
    for i, t in enumerate(toks):
        if i > 0 and CAP_RE.match(t) and t.lower() not in {'i'}:
            cap_counts[t.lower()] += 1
    repeated_entity_mentions = sum(c for c in cap_counts.values() if c >= 2)
    pronouns = sum(1 for t in low if t in PRONOUNS)
    state_transfer = sum(1 for t in low if t in STATE_TRANSFER)
    w.entity_state_score = (repeated_entity_mentions + pronouns + state_transfer) / n

    # Physical/affordance cues
    physical = sum(1 for t in low if t in PHYSICAL)
    spatial = sum(1 for t in low if t in SPATIAL)
    w.physical_score = (physical + spatial) / n

    # Social/causal cues
    social = sum(1 for t in low if t in SOCIAL)
    causal = sum(1 for t in low if t in CAUSAL_TEMPORAL)
    w.social_causal_score = (social + causal) / n

    # Composite: union of all relation cues
    w.composite_score = (repeated_entity_mentions + pronouns + state_transfer + physical + spatial + social + causal) / n


def segment_file(path: pathlib.Path, source_name: str, target_words: int = 64) -> list[Window]:
    """Segment a source file into windows of approximately target_words."""
    windows: list[Window] = []
    lines: list[str] = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        all_lines = f.readlines()

    buf_lines: list[str] = []
    buf_words = 0
    start_line = 0
    for i, line in enumerate(all_lines):
        stripped = line.rstrip('\n')
        wc = len(stripped.split())
        if wc == 0:
            continue
        if buf_words == 0:
            start_line = i
        buf_lines.append(stripped)
        buf_words += wc
        if buf_words >= target_words:
            text = ' '.join(buf_lines)
            windows.append(Window(
                source_file=source_name,
                start_line=start_line,
                end_line=i,
                text=text,
                words=len(text.split()),
            ))
            buf_lines = []
            buf_words = 0
    # Remaining buffer: attach to last window if small, else create new
    if buf_lines:
        text = ' '.join(buf_lines)
        wc = len(text.split())
        if windows and wc < target_words // 2:
            last = windows[-1]
            combined = last.text + ' ' + text
            windows[-1] = Window(
                source_file=source_name,
                start_line=last.start_line,
                end_line=len(all_lines) - 1,
                text=combined,
                words=len(combined.split()),
            )
        else:
            windows.append(Window(
                source_file=source_name,
                start_line=start_line,
                end_line=len(all_lines) - 1,
                text=text,
                words=wc,
            ))
    return windows


def select_arm_windows(
    windows_by_source: dict[str, list[Window]],
    score_attr: str,
    fraction: float,
    top: bool,
    seed: int,
) -> list[Window]:
    """Select top or bottom fraction of windows within each source by score_attr."""
    selected: list[Window] = []
    rng = random.Random(seed)
    for source, wins in sorted(windows_by_source.items()):
        ranked = sorted(wins, key=lambda w: getattr(w, score_attr), reverse=top)
        n_select = max(1, int(len(ranked) * fraction))
        chosen = ranked[:n_select]
        # Shuffle within source to avoid positional bias
        rng.shuffle(chosen)
        selected.extend(chosen)
    return selected


def select_uniform_windows(
    windows_by_source: dict[str, list[Window]],
    fraction: float,
    seed: int,
) -> list[Window]:
    """Random fraction from each source."""
    selected: list[Window] = []
    rng = random.Random(seed)
    for source, wins in sorted(windows_by_source.items()):
        n_select = max(1, int(len(wins) * fraction))
        chosen = rng.sample(wins, n_select)
        selected.extend(chosen)
    return selected


def make_repeated_jsonl(
    windows: list[Window],
    target_exposure: int,
    epoch_seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Create JSONL records with epoch-shuffled repetitions to reach target_exposure."""
    unique_words = sum(w.words for w in windows)
    if unique_words == 0:
        raise RuntimeError("empty window selection")
    n_epochs = max(1, target_exposure // unique_words)
    # Allow partial final epoch
    records: list[dict[str, Any]] = []
    cumulative = 0
    rng = random.Random(epoch_seed)
    for epoch in range(n_epochs + 1):
        order = list(range(len(windows)))
        rng.shuffle(order)
        for idx in order:
            w = windows[idx]
            if cumulative + w.words > target_exposure:
                continue  # skip to stay at or below target
            records.append({
                'text': w.text,
                'words': w.words,
                'source_file': w.source_file,
                'start_line': w.start_line,
                'end_line': w.end_line,
                'epoch': epoch,
                'mode': 'structure_density_selected',
            })
            cumulative += w.words
            if cumulative >= target_exposure:
                break
        if cumulative >= target_exposure:
            break
    return records, cumulative


def aggregate_structure_report(windows: list[Window]) -> dict[str, Any]:
    """Report aggregate structure densities for a set of windows."""
    n = len(windows)
    total_words = sum(w.words for w in windows)
    scores = {
        'entity_state': [w.entity_state_score for w in windows],
        'physical': [w.physical_score for w in windows],
        'social_causal': [w.social_causal_score for w in windows],
        'composite': [w.composite_score for w in windows],
    }
    source_words = collections.Counter()
    for w in windows:
        source_words[w.source_file] += w.words
    return {
        'windows': n,
        'unique_words': total_words,
        'source_word_proportions': {k: v / max(1, total_words) for k, v in sorted(source_words.items())},
        'score_means': {k: sum(v) / max(1, len(v)) for k, v in scores.items()},
        'score_medians': {k: sorted(v)[len(v) // 2] if v else 0 for k, v in scores.items()},
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus_dir', default='experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_v0_1M/raw_dataset')
    p.add_argument('--out_dir', default='experiments/archive/initial_model_studies/data/structure_density_revision_157')
    p.add_argument('--target_exposure', type=int, default=10000000)
    p.add_argument('--fraction', type=float, default=0.25, help='Fraction of windows from each source per arm')
    p.add_argument('--window_target_words', type=int, default=64)
    p.add_argument('--selection_seed', type=int, default=20001)
    p.add_argument('--epoch_seed', type=int, default=20002)
    args = p.parse_args()

    corpus_dir = pathlib.Path(args.corpus_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Segment and score ──
    source_files = sorted(corpus_dir.glob('*.train.txt'))
    if not source_files:
        raise RuntimeError(f'No .train.txt files in {corpus_dir}')

    windows_by_source: dict[str, list[Window]] = {}
    all_windows: list[Window] = []
    for path in source_files:
        name = path.name
        wins = segment_file(path, name, args.window_target_words)
        for w in wins:
            score_window(w)
        windows_by_source[name] = wins
        all_windows.extend(wins)
        print(f'  {name}: {len(wins)} windows, {sum(w.words for w in wins)} words')

    total_corpus_words = sum(w.words for w in all_windows)
    print(f'Total: {len(all_windows)} windows, {total_corpus_words} words')

    # ── Select arms ──
    arms_config = {
        'high_entity_state': ('entity_state_score', True),
        'high_physical': ('physical_score', True),
        'high_social_causal': ('social_causal_score', True),
        'matched_low': ('composite_score', False),
    }

    arm_windows: dict[str, list[Window]] = {}
    for arm_name, (attr, top) in arms_config.items():
        arm_windows[arm_name] = select_arm_windows(
            windows_by_source, attr, args.fraction, top, args.selection_seed)

    arm_windows['uniform'] = select_uniform_windows(
        windows_by_source, args.fraction, args.selection_seed + 100)

    # ── Report structure densities BEFORE materializing ──
    arm_reports: dict[str, dict] = {}
    for arm_name, wins in arm_windows.items():
        arm_reports[arm_name] = aggregate_structure_report(wins)

    # ── Materialize JSONL with matched repetition ──
    arm_paths: dict[str, dict] = {}
    arm_exposure: dict[str, int] = {}
    for arm_name, wins in arm_windows.items():
        records, achieved = make_repeated_jsonl(wins, args.target_exposure, args.epoch_seed)
        path = out_dir / f'{arm_name}.jsonl'
        write_jsonl(path, records)
        arm_paths[arm_name] = {
            'path': str(path),
            'rows': len(records),
            'achieved_exposure': achieved,
            'unique_words': sum(w.words for w in wins),
            'sha256': sha256_text(path.read_text(encoding='utf-8')[:10000]),
        }
        arm_exposure[arm_name] = achieved

    # ── Validate matched design ──
    unique_budgets = {name: r['unique_words'] for name, r in arm_paths.items()}
    achieved_exposures = {name: r['achieved_exposure'] for name, r in arm_paths.items()}

    # ── Save manifest ──
    manifest = {
        'status': 'STRUCTURE_DENSITY_ARMS_READY',
        'corpus_dir': str(corpus_dir),
        'source_files': [p.name for p in source_files],
        'total_corpus_words': total_corpus_words,
        'total_windows': len(all_windows),
        'window_target_words': args.window_target_words,
        'fraction': args.fraction,
        'target_exposure': args.target_exposure,
        'selection_seed': args.selection_seed,
        'epoch_seed': args.epoch_seed,
        'arms': list(arm_paths.keys()),
        'arm_paths': arm_paths,
        'arm_unique_budgets': unique_budgets,
        'arm_achieved_exposures': achieved_exposures,
        'arm_structure_reports': arm_reports,
        'design_validation': {
            'source_proportions_matched': all(
                abs(arm_reports[a]['source_word_proportions'].get(s, 0) -
                    arm_reports['uniform']['source_word_proportions'].get(s, 0)) < 0.05
                for a in arm_reports for s in arm_reports['uniform']['source_word_proportions']
            ),
            'unique_budget_range': [min(unique_budgets.values()), max(unique_budgets.values())],
            'exposure_range': [min(achieved_exposures.values()), max(achieved_exposures.values())],
            'repetition_epochs_approx': {name: args.target_exposure / max(1, arm_paths[name]['unique_words']) for name in arm_paths},
        },
        'scientific_note': (
            'Structure-density selection within each source. '
            'All arms share source-file proportions and repetition factor. '
            'The only variable is which windows within each source were selected by structure score. '
            'Train with S1 DeBERTa-v2 12x384, baseline16k, AdamW, flat WWM.'
        ),
    }
    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': manifest['status'],
        'manifest': str(manifest_path),
        'arms': list(arm_paths.keys()),
        'unique_budgets': unique_budgets,
        'achieved_exposures': achieved_exposures,
        'structure_density_means': {a: arm_reports[a]['score_means'] for a in arm_reports},
        'source_proportions_check': {a: arm_reports[a]['source_word_proportions'] for a in arm_reports},
    }, indent=2))


if __name__ == '__main__':
    main()
