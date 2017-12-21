def runpy(filename):
    """
    run .py script
    """
    with open(filename) as source_file: exec(source_file.read())
