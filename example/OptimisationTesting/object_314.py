import json
from GACF import find_correlation_from_lists_cpp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open('NGTS_Field_NG2346-3633_pdic.json') as jsf:
        objd = json.load(jsf)

    # print objd
    # timeseries = objd['timeseries']
    # flux = objd['flux']

    periods = []
    not_moon = {}
    is_moon = 0
    for k, v in objd.iteritems():
        # print k, v
        if v is not None:
            is_moon_count = False
            for p in v:
                periods.append(p)
                if ((27.5 < p < 29.5) or
                    (13.8 < p < 15.2)) and not is_moon_count:
                    is_moon += 1
                    is_moon_count = True
                elif not is_moon_count and (k not in not_moon):
                        not_moon[k] = p


    print 'Out of {} objects, {} detected with moon periods'.format(len(objd), is_moon)

    for k, v in not_moon.iteritems():
        print k, v

    bins = np.arange(1, 100, 0.01)
    plt.hist([v[0] for v in objd.values() if v is not None], bins=bins)
    plt.title('Histogram of outputted period ({} raw objects)'.format(len(objd)))
    plt.xscale('log')
    plt.show()

    plt.hist([v for v in not_moon.values()], bins=bins)
    plt.title('Histogram of outputted period ({} pruned objects)'.format(len(not_moon)))
    plt.xscale('log')
    plt.xlim([1, 100])
    plt.show()
