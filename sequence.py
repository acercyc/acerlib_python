def movingWindow(start=0, end=10, windowSize=3, shiftSize=None):
    """
    1.0 - Acer 2017/05/14 04:07
    """
    if shiftSize is None:
        shiftSize = windowSize

    c_start = start
    c_end = c_start + windowSize

    while c_end <= end:
        yield list(range(c_start, c_end))
        c_start += shiftSize
        c_end = c_start + windowSize

