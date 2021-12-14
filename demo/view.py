from PIL import Image

import sys

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Expected an image path arg')
    exit()
  image = None
  with open(sys.argv[1], 'rb') as file:
    image = Image.frombytes('L', (20, 20), file.read()).resize((200, 200))
  image.show()
