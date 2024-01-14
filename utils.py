import re, os
import numpy as np
from urllib.parse import unquote

def read_js_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        js_contents = file.read()
    return js_contents


def combine_js_files(folder_path):
    directory = []
    codes = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            code = ""
            if file.endswith('.js'):
                file_path = os.path.join(root, file)
                file_path = os.path.normpath(file_path)
                file_path = unquote(file_path)
                try:
                    with open(file_path, 'r', encoding='latin-1') as js_file:
                        js_code = js_file.read()
                        code += js_code
            #         print(root)
                except Exception:
                    pass
            directory.append(root)
            codes.append(code)
    #                         combined_file.write(js_code + '\n')
    return codes

def test_walk(main):
    print(list(os.walk(main)))
    print('===============')
    for root, dirs, files in os.walk(main):
        print(root)
        print(dirs)
        print(files)
        print('================')

def read_data(main):
    data = []
    outer = list(os.walk(main))[0]
    dirs = outer[1]

    for dir in dirs:
        inner = f'{main}/{dir}'
        inner_dirs = list(os.walk(inner))[0][1]

        if 'scripts' in inner_dirs or 'js' in inner_dirs:
            next = 'scripts' if 'scripts' in inner_dirs else 'js'
            next_path = f'{inner}/{next}'
            code_files = list(os.walk(next_path))[0][2]

            if 'background.js' in code_files:
                full_path = f'{next_path}/background.js'
                if not os.path.isdir(full_path):
                    try:
                        code = read_js_file(full_path)
                        data.append(code)
                    except Exception:
                        pass

    return data


def preprocess_and_tokenize(data):
    token_entries = []
    for code_section in data:
        code_section = re.sub(r'//.*|/\*[\s\S]*?\*/', '', code_section)
        tokens = re.findall(r'\b\w+\b', code_section)
        tokens = [token.lower() for token in tokens]
        token_entries.append(tokens)
    return token_entries

if __name__ == '__main__':
    data = read_data()
    print(data[0])

    #test_walk(r'data/extracted/aaajajlhgmmfgmcphmjnmecpfdmopfbb-2018-08-22 14%3A42%3A47.038000-f24429e7e759dfa798f1f660ccd7c520')


