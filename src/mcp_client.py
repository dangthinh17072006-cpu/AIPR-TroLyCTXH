"""Một client MCP thật: initialize -> list_tools -> call_tool qua stdio."""
import os
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import ROOT


@asynccontextmanager
async def connect(db_path=None):
    env = os.environ.copy()
    if db_path:
        env['AIPR_DB'] = str(db_path)
    params = StdioServerParameters(command=sys.executable, args=['-m', 'src.server'],
                                   cwd=str(ROOT), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Discover từ server, không dựng danh sách tool cố định ở client.
            catalog = []
            cursor = None
            while True:
                page = await session.list_tools(cursor=cursor)
                catalog.extend(t.model_dump(by_alias=True, exclude_none=True) for t in page.tools)
                cursor = page.nextCursor
                if not cursor:
                    break
            yield session, catalog


async def call(session, name, arguments):
    import asyncio
    try:
        result = await asyncio.wait_for(session.call_tool(name, arguments), timeout=15)
        text = '\n'.join(c.text for c in result.content if getattr(c, 'type', '') == 'text')
        if result.isError and not text.startswith('Error:'):
            text = 'Error: ' + text
        return text
    except Exception as exc:
        return f'Error: MCP {type(exc).__name__}'
