"""Rebuild tables and vector figures from frozen report data; no model execution."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
DATA, FIG, TAB = BASE.parents[1] / 'results', BASE / 'figures', BASE / 'tables'
BLUE, VIOLET, GRAY, INK = '#315C8A', '#76558E', '#77818C', '#202328'
WIDTH = 160 / 25.4
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['TeX Gyre Heros'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': GRAY,
    'axes.linewidth': .6, 'pdf.fonttype': 42, 'pgf.texsystem': 'xelatex',
    'pgf.rcfonts': False, 'axes.unicode_minus': False,
    'pgf.preamble': r'\usepackage{fontspec}'
                    r'\setmainfont{TeX Gyre Heros}\setsansfont{TeX Gyre Heros}',
})

def read(name):
    with (DATA / name).open(newline='') as f:
        return list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', backend='pgf')
    plt.close(fig)

def clean(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=3, width=.6)

def write(name, content):
    (TAB / name).write_text(content, encoding='utf-8')

def tex_rows(rows):
    # Keep the final alignment terminator and rule in the same input stream.
    return '\n'.join(' & '.join(row) + r' \\' for row in rows) + '\n\\bottomrule\n'

def catalog_table():
    with (BASE/'data/research_catalog.tsv').open(newline='') as stream:
        entries=list(csv.DictReader(stream,delimiter='\t'))
    assert len(entries)==74 and len({r['id'] for r in entries})==74
    def escape(text):
        return text.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
    sections=[]
    for group in dict.fromkeys(r['group'] for r in entries):
        subset=[r for r in entries if r['group']==group]
        sections.extend([r'\Needspace{8\baselineskip}',r'\subsection*{'+escape(group)+'}',
            r'\begin{longtable}{@{}P{37mm}P{113mm}@{}}',
            r'\toprule 主题 & 关键操作、结论与状态\\\midrule\endfirsthead',
            r'\toprule 主题（续） & 关键操作、结论与状态\\\midrule\endhead',
            r'\bottomrule\endfoot'])
        for row in subset:
            left=escape(row['id']+'　'+row['title'])
            right=r'\textbf{'+escape(row['status'])+r'}。'+escape(row['intervention']+row['finding'].rstrip('。'))
            target = row['section']
            if row['id'] == 'R26':
                guide = ''
            elif target.startswith('app:'):
                guide = r'导读：附录~\ref{' + target + r'}；'
            elif target in {'sec:frontier', 'sec:principles', 'sec:guided', 'sec:lineage'}:
                guide = r'导读：第~\ref{' + target + r'}~章；'
            else:
                guide = r'导读：第~\ref{' + target + r'}~节；'
            material = 'https://github.com/Oxelra-AI/Qiushi-Engine-Babylm-Research/blob/main/research/materials/' + row['id'] + '.md'
            right += r'（' + guide + r'\href{' + material + r'}{研究材料：' + row['id'] + r'}）。'
            sections.append(left+' & '+right+r' \\[3pt]')
        sections.append(r'\end{longtable}')
    write('research_catalog.tex','\n'.join(sections)+'\n')

def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    catalog_table()
    trials = read('training_strategy_comparison.csv')
    byid = {r['experiment_id']: r for r in trials}
    parent = byid['FRONTIER_REFERENCE']
    policies = [('ORDINARY_CONTINUATION', '普通续训'),
                ('SPARSE_TARGET_ACQUISITION', '密集遮盖、稀疏监督'),
                ('DENSE_ACQUISITION', '密集遮盖、密集监督'),
                ('PRINCIPLE_GUIDED', '密集遮盖、稀疏监督＋普通输入保持')]
    # Tables follow the report language; figures are shared English assets.
    policy_figure_labels = [
        'Ordinary continuation',
        'Dense masking, sparse targets',
        'Dense masking, dense targets',
        'Dense masking, sparse targets\n+ ordinary-input preservation',
    ]
    def trial(stem, seed):
        name = f'{stem}_S{seed}'
        if stem == 'PRINCIPLE_GUIDED' and seed == 65:
            name = 'PRINCIPLE_GUIDED_REPLICATION_S65'
        return byid[name]
    # Recompute every complete Overall using its nine underlying task scores.
    keys = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS',
            'GlobalPIQA', 'SuperGLUE', 'Reading', 'AoA']
    for r in trials:
        assert abs(np.mean([float(r[k]) for k in keys]) - float(r['Overall'])) < 1e-9, r['experiment_id']
        assert r['complete'] == 'true'
    summary = [['母模型', f"{float(parent['Overall']):.4f}", '同一母模型', '86,005,295']]
    for stem, label in policies:
        a, b = trial(stem, 64), trial(stem, 65)
        summary.append([label, f"{float(a['Overall']):.4f}", f"{float(b['Overall']):.4f}",
                        f"{int(a['counted_exposure_words_reported']):,}"])
    write('strategies.tex', tex_rows(summary))
    # Full vectors are split by task family without changing evaluation weights.
    ordered = [(parent, '母模型', '—')]
    for stem, label in policies:
        for seed in [64, 65]:
            ordered.append((trial(stem, seed), label, f'620{seed}'))
    for name, cols in [('local_language.tex', ['BLiMP','Supplement','EWoK','Entity','COMPS']),
                       ('local_remaining.tex', ['GlobalPIQA','SuperGLUE','Reading','AoA','Overall'])]:
        write(name, tex_rows([[label, seed] + [f"{float(r[c]):.4f}" for c in cols]
                              for r, label, seed in ordered]))
    comparisons = [
        ('SPARSE_TARGET_ACQUISITION', 'ORDINARY_CONTINUATION', r'$(M,S)$ 减普通续训'),
        ('PRINCIPLE_GUIDED', 'ORDINARY_CONTINUATION', '完整方案减普通续训'),
        ('PRINCIPLE_GUIDED', 'SPARSE_TARGET_ACQUISITION', r'完整方案减 $(M,S)$'),
        ('PRINCIPLE_GUIDED', None, '完整方案减母模型'),
    ]
    differences = []
    for method, reference, label in comparisons:
        values = []
        for seed in [64, 65]:
            a = trial(method, seed)
            b = trial(reference, seed) if reference else parent
            delta = float(a['Overall']) - float(b['Overall'])
            assert abs(sum((float(a[k])-float(b[k]))/9 for k in keys)-delta) < 1e-9
            values.append(f'{delta:.4f}')
        differences.append([label] + values)
    write('strategy_differences.tex', tex_rows(differences))
    windows = read('relation_window_controls.csv')
    assert len(windows) == 8
    window_rows = []
    for relation, label in [('exact_repetition', '逐字重复'), ('aligned_restatement', '对齐重述')]:
        for window, window_label in [('same_window', '同窗'), ('split_window', '分窗')]:
            values = []
            for seed in [43022, 43122]:
                row = next(r for r in windows if r['relation'] == relation
                           and r['window'] == window and int(r['training_seed']) == seed)
                assert int(row['targets_per_checkpoint']) == 2732
                values.append(f"${float(row['delta_true_source_advantage_nats']):+.3f}$")
            window_rows.append([label, window_label] + values)
    write('relation_windows.tex', tex_rows(window_rows))
    natural = read('natural_restatement_transfer.csv')
    assert len(natural) == 4
    natural_rows = []
    for token_class, label in [('overlap', '目标 token 在来源中出现'),
                                ('nonoverlap', '目标 token 未在来源中出现')]:
        values = []
        for relation in ['exact_repetition', 'aligned_restatement']:
            row = next(r for r in natural if r['token_class'] == token_class and r['relation'] == relation)
            assert int(row['n_training_seeds']) == 3 and int(row['pairs_per_seed']) == 1200
            values.append(f"${float(row['delta_true_source_advantage_nats']):+.3f}"
                          rf"\pm {float(row['training_seed_sd']):.3f}$")
        natural_rows.append([label] + values)
    write('natural_restatement.tex', tex_rows(natural_rows))
    finetuning = read('finetuning_seed_comparison.csv')
    assert len(finetuning) == 6
    ft_tasks = ['boolq', 'multirc', 'rte', 'wsc', 'mrpc', 'qqp', 'mnli']
    for row in finetuning:
        assert abs(np.mean([float(row[k]) for k in ft_tasks])-float(row['SuperGLUE'])) < 1e-9
        if row['finetuning_seed'] == '42':
            assert abs(float(row['SuperGLUE'])-float(byid[row['experiment_id']]['SuperGLUE'])) < 1e-9
    ft_rows = []
    for identifier, label in [('FRONTIER_REFERENCE', '母模型'),
                               ('SPARSE_TARGET_ACQUISITION_S64', r'密集遮盖、稀疏监督 $(M,S)$'),
                               ('PRINCIPLE_GUIDED_S64', '密集遮盖、稀疏监督＋普通输入保持')]:
        ft_rows.append([label] + [f"{float(next(r for r in finetuning if r['experiment_id'] == identifier and int(r['finetuning_seed']) == seed)['SuperGLUE']):.4f}"
                                 for seed in [42, 44]])
    write('finetuning_seeds.tex', tex_rows(ft_rows))
    # Separate markers encode repeated continuations, not time or confidence intervals.
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    fig.subplots_adjust(left=.35, right=.96, bottom=.23, top=.86)
    for i, (stem, label) in enumerate(policies):
        for seed, off, marker, color in [(64,.10,'o',BLUE),(65,-.10,'^',VIOLET)]:
            ax.scatter(float(trial(stem, seed)['Overall']), 3-i+off, marker=marker,
                       s=28, color=color, label=f'Continuation seed 620{seed}' if i==0 else None, zorder=3)
    ax.axvline(float(parent['Overall']), color=GRAY, lw=.7, ls='--')
    ax.text(float(parent['Overall'])+.002, 3.49, 'Parent model', fontsize=7, ha='left', color=GRAY)
    ax.set_yticks(range(4), policy_figure_labels[::-1])
    ax.set_ylim(-.45, 3.65); ax.set_xlim(42.0,42.28)
    ax.set_xticks([42.0,42.05,42.10,42.15,42.20,42.25])
    ax.set_xlabel('Overall (mean of nine metrics)')
    ax.legend(frameon=False,ncol=2,loc='lower right',bbox_to_anchor=(1,1.01),fontsize=7)
    clean(ax); save(fig,'strategy_comparison')
    relation = read('relation_context_use.csv')
    fig, axes = plt.subplots(1,2,figsize=(WIDTH,2.85))
    fig.subplots_adjust(left=.19,right=.98,bottom=.23,top=.82,wspace=.78)
    for ax, key, sd, title in zip(axes,
        ['delta_true_source_advantage_nats','delta_unrelated_source_advantage_nats'],
        ['training_seed_sd','unrelated_training_seed_sd'],
        ['True-source advantage','Unrelated-source advantage']):
        ax.axvline(0,color=GRAY,lw=.6,ls='--')
        for y,r,col in zip([1,0],relation,[GRAY,BLUE]):
            ax.errorbar(float(r[key]),y,xerr=float(r[sd]),fmt='o',color=col,
                        capsize=3,elinewidth=.9,markersize=4.5)
        ax.set_yticks([1,0],['Exact repetition','Aligned paraphrase'])
        ax.set_ylim(-.6,1.6); ax.set_xlim(-1.15,1.15)
        ax.set_xticks([-1,-.5,0,.5,1]); ax.set_xlabel('Change in advantage (nats)')
        ax.set_title(title,loc='left',fontsize=8,pad=13)
        clean(ax)
    axes[0].text(-.34,1.18,'a',transform=axes[0].transAxes,fontweight='bold',fontsize=9)
    axes[1].text(-.29,1.18,'b',transform=axes[1].transAxes,fontweight='bold',fontsize=9)
    save(fig,'relation_context_use')
    reach=read('interface_reach.csv')
    labels=['Initial acquisition','Full-sequence continuation',
            'Static relation weight (1/17)','Interleaved supervision']
    fig,ax=plt.subplots(figsize=(WIDTH,2.8))
    fig.subplots_adjust(left=.32,right=.96,bottom=.24,top=.83)
    for i,r in enumerate(reach):
        for key,off,marker,color,label in [('familiar_accuracy',.11,'o',GRAY,'Familiar symbols'),
                    ('held_accuracy',-.11,'D',BLUE,'Unseen symbols')]:
            val=float(r[key])*100
            ax.scatter(val,3-i+off,s=25,marker=marker,color=color,label=label if i==0 else None)
            ax.annotate(f'{val:.1f}',(val,3-i+off),xytext=(5,0),textcoords='offset points',fontsize=7,va='center')
    ax.axvline(25,color=GRAY,lw=.6,ls='--')
    ax.set_xlim(20,108); ax.set_xticks([25,50,75,100]); ax.set_ylim(-.5,3.5)
    ax.set_yticks([3,2,1,0],labels);ax.set_xlabel('Four-choice accuracy (%)')
    ax.legend(frameon=False,ncol=2,loc='lower right',bbox_to_anchor=(1,1.02),fontsize=7)
    clean(ax);save(fig,'interface_reach')
    # Signed task contributions. No claim of cross-task improvement or significance.
    fig,ax=plt.subplots(figsize=(WIDTH,3.4))
    fig.subplots_adjust(left=.23,right=.95,bottom=.18,top=.86)
    names=['BLiMP','BLiMP Supplement','EWoK','Entity Tracking','COMPS','GlobalPIQA','(Super)GLUE','Reading','AoA']
    for seed,offset,marker,color in [(64,.13,'o',BLUE),(65,-.13,'^',VIOLET)]:
        r=trial('PRINCIPLE_GUIDED',seed)
        delta=[(float(r[k])-float(parent[k]))/9 for k in keys]
        assert abs(sum(delta)-(float(r['Overall'])-float(parent['Overall'])))<1e-9
        ax.scatter(delta,np.arange(9)[::-1]+offset,s=24,marker=marker,color=color,label=f'Continuation seed 620{seed}',zorder=3)
    ax.axvline(0,color=GRAY,lw=.7)
    ax.set_yticks(np.arange(9)[::-1],names);ax.set_xlim(-.09,.19)
    ax.set_xlabel('Contribution to Overall (task-score change / 9)')
    ax.legend(frameon=False,ncol=2,loc='lower right',bbox_to_anchor=(1,1.02),fontsize=7)
    clean(ax);save(fig,'task_contributions')
    board=read('leaderboard_comparison.csv')
    fig,ax=plt.subplots(figsize=(WIDTH,3.5))
    fig.subplots_adjust(left=.47,right=.91,bottom=.17,top=.96)
    labels=['Qiushi Engine / Principle-guided','Qiushi Engine / Frontier']+[r['display_label'] for r in board[2:]]
    for i,r in enumerate(board):
        x=float(r['Overall']);y=9-i
        color=VIOLET if i==0 else BLUE if i==1 else GRAY
        ax.scatter(x,y,s=26 if i<2 else 18,marker='D' if i==0 else 'o',color=color)
        ax.annotate(f'{x:.2f}',(x,y),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
    ax.set_yticks(range(10),labels[::-1]);ax.set_ylim(-.6,9.6)
    ax.set_xlim(40.5,42.6);ax.set_xticks([40.5,41,41.5,42,42.5]);ax.set_xlabel('Public Overall')
    clean(ax);save(fig,'leaderboard')
    boardlabels=['Qiushi Engine / Principle-guided','Qiushi Engine / Frontier']+[r['display_label'] for r in board[2:]]
    write('leaderboard.tex',tex_rows([[label,f"{float(r['Overall']):.2f}",f"{float(r['NLP']):.2f}",f"{float(r['HumanLike']):.2f}"] for r,label in zip(board,boardlabels)]))
    for name,cols in [('board_language.tex',['BLiMP','Supplement','EWoK','Entity','COMPS']),
                      ('board_remaining.tex',['GlobalPIQA','SuperGLUE','Reading','AoA'])]:
        write(name,tex_rows([[label]+[f"{float(r[k]):.2f}" for k in cols] for r,label in zip(board,boardlabels)]))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir()) if p.is_file()}
    (BASE/'build'/'data_checks.json').write_text(json.dumps({'all_nine_task_means_checked':True,'data_sha256':manifest},indent=2)+'\n')

if __name__=='__main__':
    main()
