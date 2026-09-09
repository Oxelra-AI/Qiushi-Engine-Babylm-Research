"""Build and locally verify a standalone English TeX source archive; never upload."""
import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Destination .zip outside the public research repository')
    parser.add_argument('--replace-generated', action='store_true',
                        help='Refresh an existing source archive at the explicit output path')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.suffix != '.zip':
        parser.error('Use a .zip destination outside the public research repository.')
    if output.exists() and not args.replace_generated:
        parser.error('Destination already exists; choose another filename to preserve it.')
    subprocess.run(['python3', '-B', 'validate.py'], cwd=BASE, check=True)
    main_source = (BASE / 'main.tex').read_text()
    with tempfile.TemporaryDirectory(prefix='qiushi-babylm-tex-') as directory:
        package = Path(directory)
        for child in ['chapters', 'latex', 'tables', 'figures']:
            (package / child).mkdir()
        portable_main = main_source.replace(r'\graphicspath{{../zh/figures/}}',
                                            r'\graphicspath{{figures/}}')
        portable_main = portable_main.replace(r'\bibliography{../references}',
                                             r'\bibliography{references}')
        (package / 'main.tex').write_text(portable_main, encoding='utf-8')
        chapters = re.findall(r'\\input\{(chapters/[^}]+)\}', main_source)
        figure_names, table_names = set(), set()
        for chapter in chapters:
            source = BASE / (chapter + '.tex')
            content = source.read_text()
            shutil.copyfile(source, package / (chapter + '.tex'))
            figure_names.update(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', content))
            table_names.update(re.findall(r'\\input\{(tables/[^}]+)\}', content))
        assert len(figure_names) == 8
        for table in table_names:
            shutil.copyfile(BASE / (table + '.tex'), package / (table + '.tex'))
        for figure in figure_names:
            assert Path(figure).name == figure
            shutil.copyfile(BASE.parent / 'zh/figures' / figure, package / 'figures' / figure)
        shutil.copyfile(ROOT / 'assets/qiushi-engine-logo.png', package / 'figures/qiushi-engine-logo.png')
        style = (BASE / 'latex/style.tex').read_text().replace(
            '../../assets/qiushi-engine-logo.png', 'figures/qiushi-engine-logo.png')
        (package / 'latex/style.tex').write_text(style, encoding='utf-8')
        shutil.copyfile(BASE / 'latex/authors.tex', package / 'latex/authors.tex')
        shutil.copyfile(BASE.parent / 'references.bib', package / 'references.bib')
        shutil.copyfile(BASE / 'build/main.bbl', package / 'main.bbl')
        # Capture only the required sources before creating any build outputs.
        files = sorted(path for path in package.rglob('*') if path.is_file())
        for _ in range(3):
            result = subprocess.run(['xelatex', '-no-shell-escape', '-interaction=nonstopmode',
                                     '-halt-on-error', 'main.tex'], cwd=package,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError('Standalone build failed:\n' + result.stdout[-6000:])
        log = (package / 'main.log').read_text()
        for marker in ['Overfull', 'Missing character:', 'Font Warning:', 'undefined references',
                       'undefined citations', 'There were undefined']:
            assert marker not in log, marker
        standalone = subprocess.check_output(['pdftotext', '-layout', str(package / 'main.pdf'), '-'], text=True)
        original = (BASE / 'build/main.txt').read_text()
        assert standalone.split() == original.split(), 'Standalone PDF text differs from the report.'
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, 'w' if args.replace_generated else 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(package).as_posix())
        with zipfile.ZipFile(output) as archive:
            assert archive.testzip() is None
        print(f'Created {output}: {len(files)} source files; independent XeLaTeX build and PDF-text comparison passed.')
        print('No upload or submission was performed.')


if __name__ == '__main__':
    main()
