import re
from NGTS_Field import Field

if __name__ == '__main__':
    with open('JamesMcWork/lcs.txt', 'r') as f:
        lcstr = f.read()

    files = lcstr.split("\n")

    filepattern = re.compile(r'^(NG\d+[+-]\d+)_(\d+)\.lc\.gz$')

    matches = []

    for f in files:
        try:
            matches.append(filepattern.search(f))
        except:
            continue

    fieldname = matches[0].group(1)
    objs = []
    for m in matches:
        try:
            objs.append(int(m.group(2)))
        except:
            continue

    field = Field(fieldname, object_list=objs, log_to_console=True, root_file='/data/jtb34')

    print field
    print field.objects

    field()
