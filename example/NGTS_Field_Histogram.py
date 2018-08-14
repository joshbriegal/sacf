import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats, integrate
from NGTS_Field import return_field_from_json_str

if __name__ == "__main__":
    with open('NG2331-3922/field_reduced.json', 'r') as f:
        field = return_field_from_json_str(f.read())

    all_periods = []
    for obj in field:
        all_periods = all_periods + obj.periods

    sns.set(color_codes=True)

    sns.distplot(all_periods)
