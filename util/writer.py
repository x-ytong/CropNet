import os



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_txt(text: str, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('-' * 100 + '\n' + text + '\n' + '-' * 100)


def append_txt(text: str, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
