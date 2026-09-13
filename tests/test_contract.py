"""Offline checks of the archived notebook's actual functions; no API/model calls."""
import ast
import csv
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / 'RAG_Chatbot.ipynb').read_text())

def functions(names, **env):
    nodes = []
    for cell in NB['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(env)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook functions>', 'exec'), scope)
    return scope

class ContractChecks(unittest.TestCase):
    def test_dataset_contract(self):
        with (ROOT / 'data/dataset.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 726)
        self.assertEqual(len({r['doc_id'] for r in rows}), 726)
        self.assertEqual(len({r['category'] for r in rows}), 31)
        self.assertTrue(all(r['content'] and r['source'] for r in rows))

    def test_notebook_code_parses(self):
        for c in NB['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']))
                self.assertEqual(c['outputs'], [])

    def test_no_key_followup_preserves_previous_question(self):
        scope = functions({'msg_text', 'condense_query'}, OPENAI_API_KEY='')
        query = scope['condense_query']
        self.assertEqual(query('시설 기준은?', []), '시설 기준은?')
        history = [{'role': 'user', 'content': [{'text': '미용실 창업'}]},
                   {'role': 'assistant', 'content': '응답'}]
        self.assertEqual(query('시설 기준은?', history), '미용실 창업 시설 기준은?')

    def test_generation_history_is_bounded_and_strips_source_footer(self):
        scope = functions({'msg_text', 'build_history_messages'})
        history = [{'role': 'user', 'content': '이전 질문'}] * 7
        history.append({'role': 'assistant', 'content': '답변📚출처⚠️고지'})
        result = scope['build_history_messages'](history)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[-1]['content'], '답변')

if __name__ == '__main__':
    unittest.main()
