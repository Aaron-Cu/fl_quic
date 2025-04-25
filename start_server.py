import asyncio
from fl_quic_transport_flower import FLQuicServer

async def main():
    server = FLQuicServer()
    await server.run()

asyncio.run(main())