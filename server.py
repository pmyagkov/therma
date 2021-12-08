#!/usr/bin/env python

from array import array
import asyncio
from os import statvfs
import sys
import base64
import io
import json
import websockets
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import cv2
import scipy.misc

image_shape = (24, 32)

# plt.ion() # enables interactive plotting
fig, ax = plt.subplots(figsize=(5, 3))
canvas = ax.imshow(np.zeros(image_shape), vmin=0, vmax=60) # start plot with zeros
# cbar = fig.colorbar(image) # setup colorbar for temps
# cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)

def get_current_time_str():
  # datetime object containing current date and time
  now = datetime.now()
# dd/mm/YY H:M:S
dt_string = datetime.now().strftime("%H:%M:%S")
print("date and time =", dt_string)

def map_matrix(arr, func, next_arr=None, **kwargs):
  for i in range(0, arr.shape[0]):
    for j in range(0, arr.shape[1]):
      next_value = func(arr[i][j])
      next_arr[i][j] = next_value

      if next_arr is not None:
        next_arr[i][j] = next_value
      else:
        arr[i][j] = next_value

def tempreture2grayscale(tempr_array, **kwargs):
  max = np.max(tempr_array)
  min = np.min(tempr_array)
  rang = max - min

  image_data = np.ndarray((tempr_array.shape[0], tempr_array.shape[1]), dtype=np.uint8)
  map_matrix(tempr_array, lambda x: 255 - 255 * x / rang, image_data)

  return image_data

def process_components(components, image, **kwargs):
  def write(smth):
    if 'writer' in kwargs:
      kwargs['writer'].write((smth + '\n').encode('utf-8'))

  (labels_count, labels, stats, centroids) = components
  for i in range(0, labels_count):
  # if this is the first component then we examine the
  # *background* (typically we would just ignore this
  # component in our loop)
    if i == 0:
      text = "examining component {}/{} (background)".format(
        i + 1, labels_count)
    # otherwise, we are examining an actual connected component
    else:
      text = "examining component {}/{}".format(i + 1, labels_count)
    # print a status message update for the current connected
    # component
    write("[INFO] {}".format(text))
    # extract the connected component statistics and centroid for
    # the current label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)

    write("(x={},y={})|(w={},h={})|area={}|(cX={},cY={})".format(
          x, y, w, h, area, cX, cY))


def create_image(line, **kwargs):
  tempr_array = np.array(line.split(" ")).astype(np.uint8)
  tempr_array = np.reshape(tempr_array, image_shape)
  image_data = tempreture2grayscale(tempr_array, writer=kwargs['writer'])
  np.vectorize(lambda x: [x, 0, 0])
  image = cv2.threshold(image_data, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

  cv2.imwrite('./img/temp.jpg', image)

  tempr_color_array = np.ndarray((tempr_array.shape[0], tempr_array.shape[1], 3), dtype=np.uint8)
  max = np.max(tempr_array)
  min = np.min(tempr_array)
  diff = max - min
  map_matrix(tempr_array, lambda x: [0, 0, x / diff * 255], tempr_color_array, writer=kwargs['writer'])

  cv2.imwrite('./img/temp2.jpg', tempr_color_array)
  rgb = cv2.imread('./img/temp2.jpg')

  components = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)
  process_components(components, rgb)
  cv2.imwrite('./img/temp3.jpg', rgb)

  image_data = np.array(image)
  canvas.set_data(image_data)
  # print(components)
  # image.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))

  time_str = datetime.now().strftime("%H:%M:%S")

  ax.text(35, 26, time_str, bbox={'facecolor': '#e77c63', 'alpha': 0.5, 'pad': 3})

  plt.pause(0.001)

  buffer = io.BytesIO()
  fig.savefig(buffer, dpi=300, facecolor='#FCFCFC', bbox_inches='tight')
  buffer.seek(0)

  return (buffer, components)


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

def writeline(smth, kwargs):
  if 'writer' in kwargs:
    kwargs['writer'].write(smth.encode('utf-8'))
    kwargs['writer'].write(b'\n')


async def init_pipe():
  reader, writer = await create_stdin_reader()
  while True:
    line = await reader.read(24 * 32 * 4)
    if not line:
        break
    # writer.write(bytes(line.decode('utf-8').strip(), 'utf-8'))

    (image_buffer, components) = create_image(line.decode('utf-8').strip(), writer=writer)
    (labels_count, labels, stats, centroids) = components

    #process_components(components, writer=writer)

#    writer.write(stats)
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
