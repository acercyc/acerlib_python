def ExtByIndex(list_in, ind):
    '''
    Extract list elements by index
    list_in: to be extracted list
    ind: index (list)
    Example:
    o = ExtByIndex(list_in, [0,2,1,3])
    '''
    list_out = []
    for d in ind:
        list_out.append(list_in[d])
    return list_out


