import asyncio
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app import app
from backend import core

class ApiChecks(unittest.TestCase):
    def setUp(self):
        app.state.lock = asyncio.Lock()
        self.client = TestClient(app)
        self.env = patch.dict(os.environ, {'DEMO_ACCESS_CODE': ''})
        self.env.start()
        self.key = patch.object(core, 'OPENAI_API_KEY', '')
        self.key.start()

    def tearDown(self):
        self.env.stop(); self.key.stop()

    def test_rejects_invalid_payloads(self):
        for payload in [{'question':' '}, {'question':'x'*1001},
                        {'question':'hi','history':[{'role':'system','content':'bad'}]},
                        {'question':'hi','history':[{'role':'user','content':'x'}]*7}]:
            self.assertEqual(self.client.post('/api/chat',json=payload).status_code,422)

    def test_response_and_history_contract(self):
        hit={'doc_id':1,'title':'문서','category':'세무','source':'기관','score':.7,'chunk':'근거'}
        with patch.object(core,'rag_answer',return_value=('답변',[hit,hit])) as answer:
            response=self.client.post('/api/chat',json={'question':' 질문 ','history':[{'role':'user','content':'이전'}]})
            self.assertEqual(response.status_code,200)
            self.assertEqual(len(response.json()['sources']),1)
            answer.assert_called_once_with('질문',history=[{'role':'user','content':'이전'}])

    def test_key_mode_requires_access_code(self):
        with patch.object(core,'OPENAI_API_KEY','test-key'):
            self.assertEqual(self.client.post('/api/chat',json={'question':'질문'}).status_code,503)
        with patch.dict(os.environ,{'DEMO_ACCESS_CODE':'test-code'}):
            self.assertEqual(self.client.post('/api/chat',json={'question':'질문'}).status_code,401)

    def test_backend_error_does_not_expose_exception(self):
        with patch.object(core,'rag_answer',side_effect=RuntimeError('private detail')), patch('backend.app.logger.exception'):
            response=self.client.post('/api/chat',json={'question':'질문'})
        self.assertEqual(response.status_code,503)
        self.assertNotIn('private detail',response.text)

    def test_busy_response(self):
        class Busy:
            def locked(self): return True
        app.state.lock=Busy()
        self.assertEqual(self.client.post('/api/chat',json={'question':'질문'}).status_code,429)

if __name__=='__main__': unittest.main()
