import os


def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)