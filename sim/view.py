from PIL import Image

import sys

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Expected an image path arg')
    exit()
  image = None
  with open(sys.argv[1], 'rb') as file:
    image = Image.frombytes('L', (300, 300), file.read())
  image.show()
