import asyncio

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientOSError
from lmxy import aretry

with open(r'D:/Downloads/files-to-index.txt', encoding='utf-8') as f:
    ts = sorted({i.strip() for i in f} - {''})

host = 'http://bs03:8088'
base = 'https://aisearch-wip.wss.local:8081'
rt = aretry(ClientOSError, max_attempts=10, override_defaults=True)


async def rm() -> None:
    async with (
        ClientSession(timeout=None) as s,
        s.post(
            host + '/documents/delete', json={'file_ids': ts}, timeout=None
        ) as rsp,
    ):
        await rsp.read()


asyncio.run(rm())
