#!/usr/bin/env python

import asyncio
import sys
import base64
import io
import json
import websockets
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

image_shape = (24, 32)

# plt.ion() # enables interactive plotting
fig, ax = plt.subplots(figsize=(5, 3))
image = ax.imshow(np.zeros(image_shape), vmin=0, vmax=60) # start plot with zeros
cbar = fig.colorbar(image) # setup colorbar for temps
cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)

def get_current_time_str():
  # datetime object containing current date and time
  now = datetime.now()
# dd/mm/YY H:M:S
dt_string = datetime.now().strftime("%H:%M:%S")
print("date and time =", dt_string)

def create_image(line):

  data_array = np.array(line.split(" ")).astype(np.int32)
  data_array = (np.reshape(data_array, image_shape))
  image.set_data(np.fliplr(data_array))
  image.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))

  time_str = datetime.now().strftime("%H:%M:%S")

  ax.text(35, 26, time_str, bbox={'facecolor': '#e77c63', 'alpha': 0.5, 'pad': 3})

  plt.pause(0.001)

  buffer = io.BytesIO()
  fig.savefig(buffer, dpi=300, facecolor='#FCFCFC', bbox_inches='tight')
  buffer.seek(0)

  return buffer


open_sockets = []

async def handler(websocket):

  open_sockets.append(websocket)

  await websocket.wait_closed()
  open_sockets.remove(websocket)


  """ async for message in websocket:
    await process(message)

  while True:
    try:
      message = await websocket.recv()
      print(message)
    except websockets.ConnectionClosedOK:
      open_sockets.remove(websocket)
      break """


async def init_sockets():
  async with websockets.serve(handler, "localhost", 8001):
    await asyncio.Future()  # run forever


async def create_stdin_reader():
  loop = asyncio.get_event_loop()
  reader = asyncio.StreamReader()
  protocol = asyncio.StreamReaderProtocol(reader)

  await loop.connect_read_pipe(lambda: protocol, sys.stdin)
  w_transport, w_protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
  writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)
  return reader, writer


async def init_pipe():
  reader, writer = await create_stdin_reader()
  while True:
    line = await reader.read(24 * 32 * 4)
    if not line:
        break
    # writer.write(bytes(line.decode('utf-8').strip(), 'utf-8'))

    image_buffer = create_image(line.decode('utf-8').strip())
    image_b64 = base64.b64encode(image_buffer.getvalue())
    message = {
      "type": 'image',
      "data": {
        "image": 'data:image/jpg;base64,' + image_b64.decode('utf-8')
      }
    }

    for s in open_sockets:
      try:
        await s.send(json.dumps(message))
      except:
        open_sockets.remove(s)

    # writer.write(line)


async def main():
  await asyncio.gather(
    asyncio.create_task(init_sockets()),
    asyncio.create_task(init_pipe())
  )

if __name__ == "__main__":
    asyncio.run(main())
