"""Test navigation coverage and empty-directory maintenance without experiments."""
import json
import tempfile
import unittest
from pathlib import Path

import build_navigation as navigation
import prune_empty_directories as cleanup


class NavigationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def record(self, name, status='historical'):
        path = 'experiments/archive/functional_learning/' + name
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# Original question\n\nUnchanged scientific content.\n')
        return {'path': path, 'kind': 'note', 'collection': 'functional_learning',
                'scientific_status': status}

    def test_all_documents_once_in_machine_index(self):
        records = [self.record('notes/question.md'), self.record('plans/test.md', 'proposal'),
                   self.record('analysis/correction.md', 'superseded'),
                   self.record('data/eval/README.md')]
        outputs = navigation.render_documents(self.root, records, [])
        indexed = json.loads(outputs['research/document_index.json'])['documents']
        self.assertEqual({r['path'] for r in indexed}, {r['path'] for r in records})
        self.assertEqual(len(indexed), len(records))
        self.assertEqual([r['role'] for r in indexed].count('notes'), 2)
        self.assertIn('Superseded', outputs['research/notes/functional_learning/README.md'])
        self.assertIn('Research proposal', outputs['research/plans/functional_learning/README.md'])

    def test_research_documents_live_under_their_reading_directories(self):
        for role in ('notes', 'plans', 'documents'):
            path = f'research/{role}/functional_learning/example.md'
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# Scientific record\n')
            record = {'path': path, 'kind': 'note', 'collection': 'functional_learning',
                      'scientific_status': 'historical'}
            self.assertEqual(navigation.document_role(path), role)
            outputs = navigation.render_documents(self.root, [record], [])
            self.assertIn('(example.md)', outputs[f'research/{role}/functional_learning/README.md'])

    def test_missing_and_duplicate_records_fail(self):
        record = self.record('notes/question.md')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            navigation.document_rows(self.root, [record, record], [])
        (self.root / record['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            navigation.document_rows(self.root, [record], [])

    def test_topics_preserve_status_and_all_links(self):
        record = self.record('notes/retracted.md', 'withdrawn')
        topics = [{'id': 'R36', 'title': 'Corrected study', 'research_family': 'Memory',
                   'status': 'Withdrawn', 'explanation': 'Query-independent answers.',
                   'report_section': 'sec:memory', 'materials': [record['path']]}]
        outputs = navigation.render_topics([record], topics)
        self.assertIn('Withdrawn', outputs['research/materials/R36.md'])
        self.assertIn('retracted.md', outputs['research/materials/R36.md'])
        self.assertIn('<a id="r36"></a>', outputs['research/materials.md'])
        self.assertNotIn('\n<a ', outputs['research/materials.md'])

    def test_topic_material_must_be_in_manifest(self):
        topic = {'id': 'R01', 'title': 'Example', 'research_family': 'Models',
                 'status': 'Historical', 'explanation': 'Observed.',
                 'report_section': 'sec:example', 'materials': ['absent.md']}
        with self.assertRaisesRegex(ValueError, 'absent from manifest'):
            navigation.render_topics([], [topic])

    def test_topic_specific_reading_guide_survives_regeneration(self):
        topic = {'id': 'R01', 'title': 'Measurement example',
                 'research_family': 'Measurement', 'status': 'Conditional result',
                 'explanation': 'A bounded comparison.', 'materials': [],
                 'report_section': 'app:evaluation',
                 'report_guide': 'Report appendix: `app:evaluation`. Detailed results are listed below.'}
        text = navigation.render_topics([], [topic])['research/materials/R01.md']
        self.assertIn(topic['report_guide'], text)
        self.assertNotIn('Report section:', text)

    def test_local_only_inputs_have_acquisition_guidance_not_broken_links(self):
        record = {'path': 'experiments/archive/example/data/original.jsonl',
                  'kind': 'input', 'distribution': 'local_only'}
        topic = {'id': 'R01', 'title': 'Example', 'research_family': 'Models',
                 'status': 'Historical', 'explanation': 'Observed.',
                 'report_section': 'sec:example', 'materials': [record['path']]}
        text = navigation.render_topics([record], [topic])['research/materials/R01.md']
        self.assertIn('`' + record['path'] + '`', text)
        self.assertIn('data/README.md', text)
        self.assertNotIn('[original.jsonl]', text)

    def test_markdown_labels_and_paths_are_escaped(self):
        rendered = navigation.link('a | [b] <c>', 'research/notes/README.md',
                                   'experiments/a space(b).md')
        self.assertIn('&#124;', rendered)
        self.assertIn('%20', rendered)
        self.assertIn('%28', rendered)
        self.assertNotIn('<c>', rendered)

    def test_empty_cleanup_preserves_files_links_and_other_areas(self):
        archive = self.root / 'experiments/archive'
        (archive / 'empty/nested').mkdir(parents=True)
        (archive / 'real').mkdir()
        (archive / 'real/zero.jsonl').touch()
        (archive / 'link-holder').mkdir()
        (archive / 'link-holder/ref').symlink_to(archive / 'real')
        (self.root / 'reports/empty').mkdir(parents=True)
        dry = cleanup.prune(self.root)
        self.assertEqual(dry['empty_directories'], 2)
        self.assertTrue((archive / 'empty/nested').exists())
        applied = cleanup.prune(self.root, apply=True)
        self.assertEqual(applied['removed'], 2)
        self.assertTrue((archive / 'real/zero.jsonl').is_file())
        self.assertTrue((archive / 'link-holder/ref').is_symlink())
        self.assertTrue((self.root / 'reports/empty').is_dir())
        self.assertEqual(cleanup.prune(self.root)['empty_directories'], 0)

    def test_cleanup_does_not_follow_archive_symlink(self):
        (self.root / 'actual/empty').mkdir(parents=True)
        (self.root / 'experiments').mkdir()
        (self.root / 'experiments/archive').symlink_to(self.root / 'actual')
        self.assertEqual(cleanup.prune(self.root, apply=True)['removed'], 0)
        self.assertTrue((self.root / 'actual/empty').is_dir())

    def test_cleanup_preserves_git_and_build_directories(self):
        archive = self.root / 'experiments/archive/example'
        (archive / '.git/objects').mkdir(parents=True)
        (archive / 'build/empty').mkdir(parents=True)
        self.assertEqual(cleanup.prune(self.root, apply=True)['removed'], 0)
        self.assertTrue((archive / '.git/objects').is_dir())

    def test_cleanup_does_not_follow_external_ancestor(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (target / 'archive/empty').mkdir(parents=True)
            (self.root / 'experiments').symlink_to(target)
            self.assertEqual(cleanup.prune(self.root, apply=True)['removed'], 0)
            self.assertTrue((target / 'archive/empty').is_dir())


if __name__ == '__main__':
    unittest.main()
