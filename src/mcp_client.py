"""Client MCP: initialize -> list_tools -> call_tool qua stdio."""
import asyncio
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCP_TIMEOUT_S, ROOT


@asynccontextmanager
async def connect(db_path=None):
    env = os.environ.copy()
    if db_path:
        env["AIPR_DB"] = str(db_path)  # eval dùng DB tạm riêng
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.server"],
                                   cwd=str(ROOT), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # danh sách tool lấy từ server, không khai báo cứng ở client
            result = await session.list_tools()
            catalog = [t.model_dump(by_alias=True, exclude_none=True) for t in result.tools]
            yield session, catalog


async def call(session, name, arguments):
    """Không raise: lỗi nào cũng trả về chuỗi bắt đầu bằng 'Error:' (yêu cầu 2.6)."""
    try:
        result = await asyncio.wait_for(session.call_tool(name, arguments), timeout=MCP_TIMEOUT_S)
        text = "\n".join(c.text for c in result.content if c.type == "text")
        if result.isError and not text.startswith("Error:"):
            text = "Error: " + text
        return text
    except Exception as exc:
        return f"Error: MCP {type(exc).__name__}: {exc}"[:300]
