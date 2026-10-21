import asyncio

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientOSError
from glow import amap
from lmxy import aretry
from tqdm.auto import tqdm

with open(r'D:/Downloads/files-to-index.txt', encoding='utf-8') as f:
    ts = sorted({i.strip() for i in f} - {''})

host = 'http://bs03:8088'
base = 'https://aisearch-wip.wss.local:8081'
rt = aretry(ClientOSError, max_attempts=10, override_defaults=True)


async def add(i: str) -> None:
    async with (
        ClientSession(timeout=None) as s,
        s.put(
            host + '/file-by-id',
            json={'id': i, 'base_url': base},
            timeout=None,
        ) as rsp,
    ):
        lines = (await rsp.read()).decode().splitlines()
        *_, msg = '', *(x2 for x in lines if (x2 := x.strip()))
        tqdm.write(f'{i}: {rsp.status}: {msg}')


async def main() -> None:
    with tqdm(total=len(ts)) as pb:
        async for _ in amap(rt(add), ts, limit=16):
            pb.update()


asyncio.run(main())
