import argparse
import asyncio
import os
from dataclasses import asdict
from datetime import datetime, timezone
from .config import ROOT, load_env
from .mcp_client import connect
from .logging_utils import RunLog
from .agent import run_agent


async def main():
    load_env()
    parser=argparse.ArgumentParser(description='Trợ lý CTXH - dữ liệu giả lập')
    parser.add_argument('command',choices=['demo','chat','eval'])
    parser.add_argument('--engine',choices=['local','api'],default=os.getenv('AIPR_MODE','local'))
    parser.add_argument('--retrieval',choices=['dense','hybrid'],default='hybrid')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.engine=='api' and not os.getenv('OPENAI_API_KEY'):
        parser.error('API mode cần key trong .env; không tự chuyển sang local.')
    if args.command=='eval':
        from .evaluate import evaluate
        if args.repeats<1:parser.error('repeats >= 1')
        await evaluate(args.engine,args.repeats)
        return
    print(f'CTXH DEMO | engine={args.engine} | retrieval={args.retrieval}')
    print('Đây là dữ liệu giả lập. Mã sinh viên dùng thử: 11110001. Gõ exit để thoát.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log=RunLog(ROOT/'logs'/f'{args.command}-{args.engine}-{stamp}.jsonl')
    questions=['Tính tổng ngày CTXH của A01 và A02.',
               'Đăng ký A01 cho mã sinh viên demo 11110001.',
               'Học phí học kỳ này bao nhiêu?']
    async with connect() as (session,catalog):
        index=0
        while True:
            if args.command=='chat':
                question=input('Bạn: ').strip()
                if question.lower() in ['exit','quit']:break
                if not question:continue
            else:
                if index>=len(questions):break
                question=questions[index]
                print('\nBạn:',question)
            index+=1;log.question_id=f'D{index:02d}'
            log('QUESTION',question=question,retrieval_mode=args.retrieval)
            answer,usage,_=await run_agent(session,catalog,question,args.retrieval,args.engine,log)
            print('Trợ lý:',answer.text)
            sources=sorted({f'{s.doc_id} v{s.version} [{s.chunk_id}]' for c in answer.claims for s in c.sources})
            if sources:print('Nguồn:', '; '.join(sources))
            print('USAGE:',asdict(usage))
    print('Log:',log.path)


if __name__=='__main__':
    asyncio.run(main())
