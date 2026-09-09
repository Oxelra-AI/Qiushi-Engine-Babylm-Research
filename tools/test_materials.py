"""Check artifact selection and hashing without running research programs."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import inspect_materials
import package_release
import validate_repository


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for module in (common, inspect_materials, validate_repository):
            scope = patch.object(module, 'ROOT', self.root)
            scope.start()
            self.addCleanup(scope.stop)

    def manifest(self, entries):
        records = []
        for relative, content, distribution in entries:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            records.append({'path': relative, 'bytes': len(content),
                            'sha256': hashlib.sha256(content).hexdigest(),
                            'kind': 'input', 'distribution': distribution})
        directory = self.root / 'evidence'
        directory.mkdir(exist_ok=True)
        (directory / 'materials_manifest.json').write_text(json.dumps({'files': records}))

    def test_source_and_weight_selection(self):
        self.manifest([
            ('experiments/example.csv', b'value\n1\n', 'public'),
            ('experiments/original.jsonl', b'{"text":"sample"}\n', 'local_only'),
            ('experiments/model.safetensors', b'weight-fixture', 'public'),
        ])
        source = {str(p.relative_to(self.root)) for p in common.public_files()}
        full = {str(p.relative_to(self.root)) for p in common.public_files(include_weights=True)}
        self.assertIn('experiments/example.csv', source)
        self.assertNotIn('experiments/original.jsonl', full)
        self.assertNotIn('experiments/model.safetensors', source)
        self.assertIn('experiments/model.safetensors', full)
        self.assertEqual(inspect_materials.verify_materials(), {'input': 3})
        self.assertEqual(inspect_materials.verify_materials(public_only=True), {'input': 1})

    def test_missing_and_changed_files_fail(self):
        self.manifest([('experiments/example.csv', b'1', 'public')])
        path = self.root / 'experiments/example.csv'
        path.write_bytes(b'2')
        with self.assertRaisesRegex(ValueError, 'differs'):
            inspect_materials.verify_materials()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            inspect_materials.verify_materials()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            inspect_materials.verify_materials(public_only=True)

    def test_report_and_rights_review_are_required(self):
        self.manifest([])
        review = {'status': 'approved', 'content_review_complete': True,
                  'source_fidelity_review_complete': True, 'coverage_review_complete': True}
        path = self.root / 'evidence/release_review.json'
        path.write_text(json.dumps(review))
        with self.assertRaises(SystemExit):
            package_release.require_release_review(self.root)
        review.update(report_alignment_complete=True, redistribution_review_complete=True)
        path.write_text(json.dumps(review))
        package_release.require_release_review(self.root)

    def test_short_private_alias_in_metadata_is_rejected(self):
        path = self.root / 'experiments/provenance.json'
        text = '{"source": "prefix_' + 's' + '0101' + '_suffix"}'
        with self.assertRaisesRegex(ValueError, 'Private material'):
            validate_repository.inspect_text(path, text)
        validate_repository.inspect_text(path, '{"source": "initial_model_studies"}')

    def test_public_prose_is_english_and_keeps_scientific_examples(self):
        path = self.root/'research/notes/example.md'
        with self.assertRaisesRegex(ValueError, 'Non-English public prose'):
            validate_repository.inspect_text(path, '# \u7814\u7a76\u7b14\u8bb0\n')
        validate_repository.inspect_text(path, '# Tokenizer analysis\nOriginal character: `\u7834`.\n')
        validate_repository.inspect_text(path, '# AoA\nHuman-like evaluation uses child acquisition norms.\n')
        validate_repository.inspect_text(self.root/'reports/zh/README.md', '# \u7814\u7a76\u62a5\u544a\n')

    def test_operational_headings_are_not_public_research(self):
        path = self.root/'research/notes/example.md'
        with self.assertRaisesRegex(ValueError, 'Operational wording'):
            validate_repository.inspect_text(path, '# Six-Session Research Synthesis\n')

    def test_internal_runtime_wording_and_abbreviated_host_path(self):
        path = self.root/'research/notes/example.md'
        for qualifier in ['', 'AI ', 'Python ']:
            with self.subTest(qualifier=qualifier):
                with self.assertRaisesRegex(ValueError, 'Operational wording'):
                    validate_repository.inspect_text(path, 'Uses the Qiushi '+qualifier+'runtime.\n')
        with self.assertRaisesRegex(ValueError, 'Private material'):
            validate_repository.inspect_text(self.root/'experiments/example.py', '# .../users'+'/admin\n')

    def test_quoted_evaluation_context_is_not_personnel_provenance(self):
        path = self.root/'experiments/example.json'
        validate_repository.inspect_text(path, json.dumps({
            'context': 'A human-in-the-loop security framework is discussed.',
            'surprisal': 1.25,
        }))


if __name__ == '__main__':
    unittest.main()
