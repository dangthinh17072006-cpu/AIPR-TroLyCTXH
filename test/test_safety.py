import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.agent import parse_action, run_agent
from src.models import Action, Claim, Source
from src.retrieval import Retriever
from src.evaluate import supported
from src.mcp_client import connect, call

class ParserTests(unittest.TestCase):
    def test_invalid_json_is_error(self):
        for value in ['{','not json','[]','{"tool":4}','{"tool":"x","arguments":[]}']:
            self.assertTrue(parse_action(value).startswith('Error:'))
    def test_json_fence(self):
        self.assertEqual(parse_action('```json\n{"tool":"finish"}\n```').tool,'finish')
    def test_version_and_owner_filter(self):
        retriever=Retriever()
        self.assertEqual(retriever.search('CTXH',owner='Other owner'),[])
        policy=next(d for d in retriever.docs if d.doc_id=='POLICY')
        self.assertEqual(policy.version,2)
    def test_faithfulness_rejects_stale_citation(self):
        claim=Claim('hours_per_day',6,[Source('POLICY',1,'POLICY:v1:0')])
        self.assertFalse(supported(claim))
    def test_faithfulness_rejects_wrong_math(self):
        claim=Claim('days',99,[Source('POLICY',2,'POLICY:v2:0'),Source('ACT-A01',1,'ACT-A01:v1:0')])
        self.assertFalse(supported(claim))

class RetrievalTests(unittest.TestCase):
    def test_chunking_never_splits_a_sentence_and_keeps_ids(self):
        from src.chunking import chunk_documents, split_sentences
        r=Retriever(); chunks=chunk_documents(r.docs)
        policy=[c for c in chunks if c.doc_id=='POLICY']
        self.assertEqual([c.chunk_id for c in policy],['POLICY:v2:0','POLICY:v2:1'])
        for c in chunks:
            self.assertTrue(all(s in c.text for s in split_sentences(c.text)))
        self.assertIn('8 giờ bằng 1 ngày',policy[0].text)
    def test_two_modes_return_chunks_and_hybrid_uses_bm25(self):
        r=Retriever()
        for mode in ('dense','hybrid'):
            hits=r.search('A03 Tiếp sức mùa thi',mode)
            self.assertEqual(hits[0]['doc_id'],'ACT-A03')
            self.assertLessEqual(len(hits),4)
        with self.assertRaises(ValueError): r.search('x','bm25')

class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_side_effect_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with connect(Path(tmp)/'db.sqlite3') as (session,catalog):
                self.assertGreaterEqual(len(catalog),2)
                events=[]
                ans,_,observed=await run_agent(session,catalog,'Đăng ký A01 cho sinh viên 11110001.',
                                               'hybrid','local',lambda e,**kw:events.append((e,kw)))
                self.assertTrue(ans.verified)
                names=[o['tool'] for o in observed]
                pos=names.index('register_activity')
                self.assertEqual(names[pos+1],'get_registration')
                row=json.loads(observed[pos]['result'])['registration']
                again=json.loads(await call(session,'register_activity',{'student_id':'11110001','activity_code':'A01'}))
                self.assertFalse(again['created'])
                self.assertEqual(row['registration_id'],again['registration']['registration_id'])
                self.assertTrue(any(e=='USAGE' for e,_ in events))
    async def test_closed_unknown_missing_hours_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with connect(Path(tmp)/'db.sqlite3') as (session,catalog):
                cases=[('register_activity',{'student_id':'11110001','activity_code':'A05'}),
                       ('register_activity',{'student_id':'bad','activity_code':'A01'}),
                       ('calculate_days',{'activity_codes':['A04']}),
                       ('calculate_days',{'activity_codes':['Z99']}),
                       ('get_registration',{'registration_id':'missing'}),
                       ('not_a_tool',{})]
                for name,args in cases:
                    result=await call(session,name,args)
                    self.assertTrue(result.startswith('Error:'),(name,result))
    async def test_capacity_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with connect(Path(tmp)/'db.sqlite3') as (session,catalog):
                for i in range(10):
                    result=await call(session,'register_activity',{'student_id':f'{11110000+i}','activity_code':'A03'})
                    self.assertFalse(result.startswith('Error:'))
                full=await call(session,'register_activity',{'student_id':'22220001','activity_code':'A03'})
                self.assertTrue(full.startswith('Error:'))
    async def test_negated_request_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with connect(Path(tmp)/'db.sqlite3') as (session,catalog):
                ans,_,obs=await run_agent(session,catalog,'Không đăng ký. Chỉ tính A01.',
                                         'hybrid','local',lambda *a,**kw:None)
                self.assertFalse(any(o['tool']=='register_activity' for o in obs))
    async def test_malformed_model_stops_at_max_steps(self):
        with patch('src.agent.call_llm',return_value='{broken'):
            events=[]
            ans,_,_=await run_agent(None,[],'A01','hybrid','api',lambda e,**kw:events.append(e))
            self.assertEqual(ans.status,'error')
            self.assertEqual(events.count('ACTION_ERROR'),8)
    async def test_model_cannot_force_unauthorized_write(self):
        action='{"tool":"register_activity","arguments":{"student_id":"11110001","activity_code":"A01"}}'
        with patch('src.agent.call_llm',return_value=action), patch('src.agent.call') as invoke:
            ans,_,_=await run_agent(None,[{'name':'register_activity'}],'Chỉ xem A01',
                                    'hybrid','api',lambda *a,**kw:None)
            invoke.assert_not_called()
            self.assertEqual(ans.status,'error')
    async def test_verify_failure_never_claims_success(self):
        from src.mcp_client import call as real_call
        async def faulty(session,name,args):
            if name=='get_registration':return 'Error: simulated read failure'
            return await real_call(session,name,args)
        with tempfile.TemporaryDirectory() as tmp:
            async with connect(Path(tmp)/'db.sqlite3') as (session,catalog):
                with patch('src.agent.call',side_effect=faulty):
                    ans,_,obs=await run_agent(session,catalog,'Đăng ký A01 cho sinh viên 11110001',
                                              'hybrid','local',lambda *a,**kw:None)
                    self.assertFalse(ans.verified)
                    self.assertEqual(ans.status,'error')
                    self.assertEqual(sum(o['tool']=='register_activity' for o in obs),1)

if __name__=='__main__':unittest.main()
