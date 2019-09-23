from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':

    hdu = fits.open('/Users/joshbriegal/GitHub/GACF/example/RunOnFieldJan19/NG2346-3633_802_2016_CYCLE1807.fits')
    obj_ids = hdu['CATALOGUE'].data['OBJ_ID']
    flux = hdu['SYSREM_FLUX3'].data
    hjd = hdu['HJD'].data
    flags = hdu['FLAGS'].data

    print 'read data'

    dic = {'OBJ_ID': np.array(obj_ids).tolist(),
           'SYSREM_FLUX3': flux.tolist(),
           'HJD': hjd.tolist(),
           'FLAGS': flags.tolist()}

    print 'compiled dic'

    with open('info_dic.json', 'w') as f:
        json.dump(dic, f)

    print 'saved'
