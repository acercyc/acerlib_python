def findX11DisplayPort():
    ''' Find localhost number for display
    1.0 - Acer 2017/02/07 15:31
    1.1 - Acer 2017/06/15 14:48
    '''
    import os
    for iPort in range(10, 30):
        try:
            portname = "localhost:%.1f" % iPort
            print('Try {}'.format(portname), end='...')
            os.environ['DISPLAY'] = portname

            import matplotlib.pyplot as plt
            f_temp = plt.figure()
            plt.close(f_temp)
            print('Succeed')
            break
        except:
            print('fail')