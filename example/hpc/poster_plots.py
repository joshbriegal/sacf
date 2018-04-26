import sys

# import seaborn as sns
import numpy as np
from scipy import stats, integrate
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.ticker as tk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import re
from copy import deepcopy
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import pandas as pd
import scipy.interpolate as interpolate
import json
import time

plt.rcParams.update({'font.size': 20})
sys.path.append('/home/jtb34/GitHub/GACF/')
#sys.path.append('/home/jtb34/python27')
# print sys.path
from NGTS.NGTS_Field import return_field_from_object_directory, NGTSField
from NGTS.GACF_utils import TIME_CONVERSIONS
import NGTS.GACF_utils as utils

from astropy.coordinates import get_moon, get_sun, SkyCoord, EarthLocation, AltAz
import astropy.units as u

from astroquery.utils.tap.core import TapPlus
from astroquery.gaia import Gaia

# ROOT_DIR = '/home/jtb34/rds/hpc-work/GACF_OUTPUTS'
ROOT_DIR = '/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS'


# XMATCH_LOCATION = '/home/jtb34/rds/hpc-work/GACF_OUTPUTS/{}/cross_match/'
XMATCH_LOCATION = '/home/jtb34/rds/rds-jtb34-gacf/GACF_OUTPUTS/{}/cross_match/'
XMATCH_FILE_NAME = 'Uncut_Final_{}.fits'
def load_gaia_params(field):
    xmatch_path = os.path.join(XMATCH_LOCATION.format(field.fieldname),
                               XMATCH_FILE_NAME.format(field.fieldname))
    if not os.path.exists(xmatch_path):
        print xmatch_path, 'does not exist'
        return field, False

    with fits.open(xmatch_path) as gfits:
        for i, obj in enumerate(gfits[1].data['Sequence_number']):
            if obj in field.objects:
                field[obj].Gaia_Teff = gfits[1].data['Gaia_Teff'][i]
                field[obj].Gaia_Radius = gfits[1].data['Gaia_Radius'][i]
                field[obj].Gaia_Lum = gfits[1].data['Gaia_Lum'][i]
                field[obj].Gaia_Parallax = gfits[1].data['Gaia_Parallax'][i]
                field[obj].TWOMASS_Hmag = gfits[1].data['2MASS_Hmag'][i]
                field[obj].TWOMASS_Kmag = gfits[1].data['2MASS_Kmag'][i]
                field[obj].APASS_Vmag = gfits[1].data['APASS_Vmag'][i]
                field[obj].APASS_Bmag = gfits[1].data['APASS_Bmag'][i]
                field[obj].Gaia_Gmag = gfits[1].data['Gaia_Gmag'][i]
                field[obj].Gaia_RPmag = gfits[1].data['Gaia_RPmag'][i]
                field[obj].Gaia_BPmag = gfits[1].data['Gaia_BPmag'][i]
                field[obj].NGTS_I_3 = gfits[1].data['NGTS_I_3'][i]
                field[obj].BminusV = field[obj].APASS_Bmag - field[obj].APASS_Vmag
                field[obj].HminusK = field[obj].TWOMASS_Hmag - field[obj].TWOMASS_Kmag
                field[obj].GminusK = field[obj].Gaia_Gmag - field[obj].TWOMASS_Kmag
                field[obj].BPminusRP = field[obj].Gaia_BPmag - field[obj].Gaia_RPmag
#         print gfits[1].header

    return field, True

def find_fits_file(directory=None):
    if directory is None:
         directory = os.getcwd()
    string_pattern = r'^(?P<fieldname>\w+?[+-]\w+?)_\d+_[\w\-/,]+_(?P<test>\w+).fits$'
    pattern = re.compile(string_pattern)
    for f in os.listdir(directory):
        match = re.match(pattern, f)
        if match is not None:
            return f if directory is None else os.path.join(directory, f), match.group('fieldname'), match.group('test')
        
    raise IOError('File not found in directory {}'.format(directory))
    
# field.load_from_fits(fits_file, obj_ids=object_list, method='fitsio')

pdics = {}
fieldnames = []
field_pattern = re.compile(r'^NG\d+[+|-]\d+$')

for f in os.listdir(ROOT_DIR):
    if re.match(field_pattern, f):
        fieldnames.append(f)
print "Found {} fields".format(len(fieldnames))
# print fieldnames
# fieldnames = fieldnames[1:5]
# fieldnames = ['NG2346-3633']

for field in fieldnames:
    field_dir = os.path.join(ROOT_DIR, field)
    pdic_filename = 'NGTS_Field_{}_pdic.json'.format(field)
    if pdic_filename in os.listdir(field_dir):
        with open(os.path.join(*[ROOT_DIR, field, pdic_filename])) as f:
            pdics[field] = json.load(f)
            pdics[field] = {int(k):v for k,v in pdics[field].iteritems()}
            

# fieldnames = pdics.keys()
# fieldnames = ['NG0004-2950']
# with open('/home/jtb34/rds/hpc-work/GACF_OUTPUTS/NG0535-0523/NGTS_Field_NG0535-0523_pdic.json', 'r') as f:
#     pdic = json.load(f)

# pdic = {int(k):v for k,v in pdic.iteritems()}


object_pattern = re.compile(r'^(?P<obj>\d+)_VERSION_CYCLE1807$')
obj_lists = {}
for field in fieldnames:
    obj_list = []
    for f in os.listdir(os.path.join(ROOT_DIR, field)):
        match = re.match(object_pattern, f)
        if match is not None:
            obj_list.append(int(match.group('obj')))
    obj_lists[field] = np.array(obj_list)
# print 'Found {} objects'.format(len(obj_list))
# print pdic

# half_moon_objs = [(k, v[0]) for k,v in pdic.iteritems() if 13<v[0]<15]
# print half_moon_objs

fieldname_pattern = re.compile('^NG(\d{2})(\d{2})([+|-])(\d{2})(\d{2})$')
fieldcentres = {}
for fieldname in fieldnames:
    match = re.match(pattern=fieldname_pattern, string=fieldname)
    if match:
        field_position = match.group(1) + 'h' + match.group(2) + 'm ' + match.group(3) + match.group(4) + ':' + match.group(5)
        coord = SkyCoord(field_position, unit=(u.hourangle, u.deg), frame='gcrs')
        fieldcentres[fieldname] = coord
# print fieldcentres

# ra_rad = [m.ra.wrap_at(180 * u.deg).radian for m in moon_sep]
# dec_rad = [m.dec.radian for m in moon_sep]
fig = plt.figure(figsize=(16.555118,16.555118))
ax = fig.add_subplot(111, projection='aitoff')
plt.title("NGTS Field Centres")
plt.grid(True)
# plt.plot(ra_rad, dec_rad, 'o', markersize=2, alpha=0.3)
zero_pos = SkyCoord('0 -0', unit=u.deg)
distances = [c.separation(zero_pos).radian for c in fieldcentres.values()]
norm = mpl.colors.Normalize(vmax=max(distances), vmin=min(distances))
# colors = plt.cm.cool_r(norm(np.linspace(min(distances), max(distances), len(distances))))
temp = np.array(distances).argsort()
ranks = np.empty_like(temp)
ranks[temp] = np.arange(len(distances))
colors = plt.cm.rainbow(ranks.astype(float) / max(ranks))
# ras = [c.ra.wrap_at(180*u.deg).radian for c in fieldcentres.values()]
# decs = [c.dec.radian for c in fieldcentres.values()]
i = 0
fieldcolors = {}
for f,c in fieldcentres.iteritems():
#     distance_from_0  = c.separation(SkyCoord(0))
#     plt.plot(c.ra.wrap_at(180*u.deg).radian, c.dec.radian, 'o', markersize=10, label=f, c=colors[i])
    plt.plot(c.transform_to('icrs').ra.wrap_at(180*u.deg).radian, c.transform_to('icrs').dec.radian, 'o', markersize=10, label=f, c=colors[i])
    fieldcolors[f] = colors[i]
    i += 1

galactic_longitudes = np.arange(start=0, stop=360, step=0.1)
galactic_latitudes = np.zeros(len(galactic_longitudes))
icrs = SkyCoord(galactic_longitudes, galactic_latitudes,
                unit="deg", frame="galactic").icrs
ra_rad = icrs.ra.wrap_at(180 * u.deg).radian
# ra_rad = icrs.ra.radian
dec_rad = icrs.dec.radian
# plt.plot(ra_rad, dec_rad, 'o', markersize=2, alpha=0.3)
plt.plot(ra_rad, dec_rad, c='b')
plt.plot(ra_rad[0], dec_rad[0], '*', c='k', markersize=30)

plt.subplots_adjust(top=0.95,bottom=0.0)
ax=plt.gca()
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*zip(*sorted(zip(zip(labels, handles), distances), key=lambda x: x[1]))[0])
# a = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# labels, handles
ax.legend(handles, labels, bbox_to_anchor=(1.04,0.5), loc="center left")

# print fieldcolors

# c = ax.scatter([m.deg for m in moon_sep], np.linspace(1,1,len(ts)),c=ts)
# plt.show()

fig.savefig("/home/jtb34/GitHub/GACF/example/hpc/Field_Position.pdf", bbox_inches='tight')

# print fieldcolors
#fieldnames = ['NG0004-2950','NG0409-1941', 'NG2331-3922']
#fieldnames = ['NG2331-3922']

fields = {}
# gfields = {}

for fieldname in fieldnames:
    t1 = time.time()
    try:
        field = return_field_from_object_directory(ROOT_DIR, fieldname, test='CYCLE1807',
                                                   obj_ids = pdics[fieldname].keys(),
                                                   include_empty_objects=False, silent=True)
    except KeyError:
        print 'No data loaded for field {}'.format(fieldname)
        continue
    t2 = time.time()
    field, ok = load_gaia_params(field)
    if field.num_objects == 0:
        ok = False
    t3 = time.time()
    for obj in field:
        try:
            obj.cleaned_refined_periods = pdics[fieldname][obj.obj]
        except KeyError:
            obj.cleaned_refined_periods = None

#     gfield = NGTSField(fieldname=fieldname, test='CYCLE1807', object_list=obj_lists[fieldname])
#     gfield = load_gaia_params(gfield)

    fields[fieldname] = field if ok else None
    t4 = time.time()
#     gfields[fieldname] = gfield

#     print '{} objects with periods, {} total'.format(field.num_objects, gfield.num_objects)
    print 'Loaded field in {} seconds'.format(t2-t1)
    print 'Loaded Gaia params in {} seconds'.format(t3-t2)
    print 'Finished Field in {} seconds'.format(t4-t1)
    
    xmatch_ids = {}

fields = {k:v for k, v in fields.iteritems() if v is not None}
fieldnames = fields.keys()
    
xmatch_paths = [os.path.join(XMATCH_LOCATION.format(fieldname),
                               XMATCH_FILE_NAME.format(fieldname)) for fieldname in fieldnames]
for xmatch_path, fieldname in zip(xmatch_paths, fieldnames):
    with fits.open(xmatch_path) as gfits:
        xmatch_ids[fieldname] = {k: v for k, v in zip(map(int,gfits[1].data['Sequence_number']), gfits[1].data['Gaia_Source_ID'])}
        
for k, v in xmatch_ids.iteritems():
    print k, len(v), 'Gaia xmatches'
# print xmatch_ids

# gaia_seq_nums = [g[1] for g in xmatch_ids if g[1] > 0]
# print len(gaia_seq_nums)
# print gaia_seq_nums

# gaia = TapPlus(url="http://gea.esac.esa.int/tap-server/tap")
gaia_ids = []
for f, v in xmatch_ids.iteritems():
    for g in v.values():
        if g > 0:
            gaia_ids.append(g)
# gaia_ids = ', '.join(map(str, gaia_ids))
# print ', '.join(map(str, gaia_ids[:5]))
job_str =  "SELECT source_id, dist.r_est, dist.r_lo, dist.r_hi, dist.r_len, \
            dist.result_flag, dist.modality_flag \n \
            FROM external.gaiadr2_geometric_distance as dist \n \
            JOIN gaiadr2.gaia_source AS src USING (source_id) \n \
            WHERE source_id IN ({})".format(', '.join(map(str, gaia_ids)))
# print job_str
job = Gaia.launch_job_async(job_str)
r = job.get_results()
# print r
distances = {k:v for k,v in zip(map(int, r['source_id'].tolist()), r['r_est'].tolist())}

for fieldname, field in fields.iteritems():
    for obj in field:
        try:
            obj.gaia_source_id = xmatch_ids[fieldname][obj.obj]
            obj.distance = distances[obj.gaia_source_id]
#             obj.distance = np.divide(1000., obj.Gaia_Parallax)
            obj.Abs_Gmag = obj.Gaia_Gmag - 5 * np.log10(obj.distance) + 5
        except (KeyError, AttributeError, Exception) as e:
    #         print '{} no distance found'.format(obj)
            obj.distance = None
            obj.Abs_Gmag = None
            
fig, ax = plt.subplots(figsize=(16.555118,16.555118*0.4))
figu, axu = plt.subplots(figsize=(16.555118,16.555118*0.4))
GminusK, AbsG, prot = [], [], []
GminusK_bkg, AbsG_bkg = [], []
GminusKu, AbsGu, protu = [], [], []
objcount = 0
pcount =  0
for fieldname, field in fields.iteritems():
    GminusK_f, AbsG_f, prot_f = [], [], []
    for obj in field:
        objcount += 1
        if obj.Abs_Gmag is not None:
            if obj.cleaned_refined_periods:
                for p in obj.cleaned_refined_periods:
                    pcount += 1
                    GminusK_f.append(obj.BPminusRP)
                    AbsG_f.append(obj.Abs_Gmag)
                    prot_f.append(p)
            else:
                GminusK_bkg.append(obj.BPminusRP)
                AbsG_bkg.append(obj.Abs_Gmag)
                
    ax.scatter(GminusK_f, prot_f, s=40, label=fieldname, c=fieldcolors[fieldname])
    prot.append(prot_f)
    GminusK.append(GminusK_f)
    AbsG.append(AbsG_f)
    
    prot_f = np.array(prot_f)
    idx_sort = np.argsort(prot_f)
    sorted_prot = prot_f[idx_sort]
    vals, idx_start, count = np.unique(sorted_prot, return_counts=True, return_index=True)
    # sets of indices
    res = np.split(idx_sort, idx_start[1:])
    #filter them with respect to their size, keeping only items occurring more than once
    vals = vals[count <= 1]
    res = filter(lambda x: x.size <= 1, res)
#     res = [x for x in res if len(x) <=1]
    idxu = np.hstack(res)  
    
# #     _, idxu, counts = np.unique(prot_f, return_index=True, return_counts=True)
#     print 'max count: {}'.format(max(count))
    prot_f = prot_f[idxu]
    GminusK_f = (np.array(GminusK_f))[idxu]
    AbsG_f = (np.array(AbsG_f))[idxu]
    protu.append(prot_f)
    GminusKu.append(GminusK_f)
    AbsGu.append(AbsG_f)
    axu.scatter(GminusK_f, prot_f, s=40, label=fieldname, c=fieldcolors[fieldname])
#     print len(np.unique(prot_f)) == len(prot_f)

prot = np.hstack(prot).flatten()
GminusK = np.hstack(GminusK).flatten()
AbsG = np.hstack(AbsG).flatten()
protu = np.hstack(protu).flatten()
GminusKu = np.hstack(GminusKu).flatten()
AbsGu = np.hstack(AbsGu).flatten()

for a in [ax, axu]:
    a.set_ylabel('Rotation Period / days')
    a.set_xlabel('Gaia BP-RP')
    a.set_yscale('log')

    a.legend(handles, labels, bbox_to_anchor=(1.04,0.5), loc="center left")

    # ax.invert_yaxis()
    #fig.legend()
    a.set_xlim([-0.2,3.5])
    a.set_ylim([1.0, 120.])
fig.subplots_adjust(top=0.95,bottom=0.0)
figu.subplots_adjust(top=0.95,bottom=0.0)
ax.set_title('Rotation Periods for {} Stars of {} total'.format(len(prot), objcount))
axu.set_title('Rotation Periods for {} Stars of {} total'.format(len(protu), objcount))
# ax.invert_xaxis()

fig.savefig("/home/jtb34/GitHub/GACF/example/hpc/Rotation_vs_BP-RP.pdf", bbox_inches='tight')
figu.savefig("/home/jtb34/GitHub/GACF/example/hpc/Rotation_vs_BP-RP_unique.pdf", bbox_inches='tight')
        
fig, ax = plt.subplots(figsize=(16.555118,16.555118))
figu, axu = plt.subplots(figsize=(16.555118,16.555118))

ax.scatter(GminusK_bkg, AbsG_bkg, s=40, c='grey', alpha=0.1)
scat = ax.scatter(GminusK, AbsG, s=40, alpha=1, c=prot, cmap='viridis', 
                  norm=mpl.colors.LogNorm(vmin=1,vmax=120))

axu.scatter(GminusK_bkg, AbsG_bkg, s=40, c='grey', alpha=0.1)
scatu = axu.scatter(GminusKu, AbsGu, s=40, alpha=1, c=protu, cmap='viridis', 
                  norm=mpl.colors.LogNorm(vmin=1,vmax=120))

for a in [ax, axu]:
    a.set_ylabel('Absolute G Magnitude')
    a.set_xlabel('Gaia BP-RP')

    a.invert_yaxis()
    a.set_xlim([-0.2,3.5])
ax.set_title('Rotation Periods for {} Stars of {} total'.format(len(prot), objcount))
axu.set_title('Rotation Periods for {} Stars of {} total'.format(len(protu), objcount))
# ax.invert_xaxis()

# sm._A = []
cb=fig.colorbar(scat, ax=ax)
cb.set_label('Rotation Period / days')
cbu=fig.colorbar(scatu, ax=axu)
cbu.set_label('Rotation Period / days')

fig.savefig("/home/jtb34/GitHub/GACF/example/hpc/HR_Diagram.pdf",bbox_inches='tight')
figu.savefig("/home/jtb34/GitHub/GACF/example/hpc/HR_Diagram_unique.pdf",bbox_inches='tight')
