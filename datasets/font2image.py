from ttf_utils import *
from tqdm import tqdm

char_list = open('./data/IDS/char_3755.txt', 'r', encoding='utf-8').read()

font_file = ''
image_file = ''
fonts = os.listdir(font_file)

for font in tqdm(fonts):
    font_path = os.path.join(font_file, font)
    print(font_path)
    try:
        font2image(font_path, image_file, char_list, 128)
        print(font)
    except Exception as e:
        print(e)
remove_empty_floder(image_file)
