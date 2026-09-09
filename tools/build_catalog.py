"""Generate the readable research catalogue from the report's shared TSV."""
import csv
from common import ROOT


def main():
    with (ROOT/'results/research_catalog.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    lines = ['# Research catalogue', '',
             'Scientific topics, interventions, findings and their current status. Identifiers are navigation aids, not a count of independent innovations.', '',
             '[Three research stages](README.md) | [Chinese report](../reports/zh/qiushi-engine-babylm-report-zh.pdf)', '']
    for group in dict.fromkeys(r['group'] for r in rows):
        lines += ['## '+group, '']
        for r in [r for r in rows if r['group'] == group]:
            lines += [f"### {r['id']} - {r['title']}", '', f"**{r['status']}.** {r['intervention']} {r['finding']}", '']
    (ROOT/'research/catalog.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
