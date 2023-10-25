import logging
import os
from typing import Callable, Dict, List, Optional, Tuple
import csv
import json, time
from collections import defaultdict
from datetime import datetime
from itertools import combinations, islice
import pickle

import pandas as pd

import torch
torch.__version__

data_df = pd.read_csv('/data/corpora_alpha/MIMIC/physionet.org/files/mimiciv/2.2/hosp/admissions.csv.gz', nrows=None, compression='gzip', 
            dtype={'subject_id': str, 'hadm_id': str},
            error_bad_lines=False)
visitid2dischargedate = {}
for ind, row in data_df.iterrows():
    visitid2dischargedate[row['hadm_id']] = row['dischtime'][0:10]

print(min(visitid2dischargedate.values()))
print(max(visitid2dischargedate.values()))


data_df = pd.read_csv('/data/corpora_alpha/MIMIC/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz', nrows=None, compression='gzip',
            dtype={'subject_id': str, 'hadm_id': str, 'icd_code': str, 'icd_version': str},
            error_bad_lines=False)
patients = defaultdict(lambda: defaultdict(list)) #lambda: "Not Present"
for ind, row in data_df.iterrows():
    hadm_id = row['hadm_id']
    scrssn = row['subject_id']
    visit_date = visitid2dischargedate[hadm_id]
    patients[scrssn][visit_date].append(row['icd_version'] +'-'+ row['icd_code']) # 687621183, 912831070              

num_icd_pat = defaultdict(int)
for k,v in patients.items():
    for kv, vv in v.items():
        for icdcode in vv:
            if icdcode.startswith("10-"):
                num_icd_pat[k] += 1
                break

print(len(patients))
print(len(num_icd_pat))
num_pos = 0
for k,v in num_icd_pat.items():
    if v > 1:
        num_pos += 1
print(num_pos)

print("Done")

def icd2cui(patients, logging_step=50000):
    dictionary = defaultdict(int)
    # cuis_li = []
    cuis_di = {}
    date_di = {}
    num_idx = 0
    for pssn,v in patients.items():
        num_idx += 1
        if num_idx%logging_step == 0:
            print("|{} - Processed {}".format(time.asctime(time.localtime(time.time())), num_idx), flush=True)
        cuis_di[pssn] = []
        cuis_li_tmp = []
        date_li_tmp = []
        for datetime_str in sorted(v.keys()): # sort by time
            datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d') # make sure time str is correct
            infos = v[datetime_str]
            if len(infos) > 0:
                # cuis_di[pssn].append((cuis, ext_cuis, strs))
                cuis_li_tmp.append((infos, [], []))
                date_li_tmp.append(datetime_str)
            for cui_id in infos:
                dictionary[cui_id] += 1
        if len(cuis_li_tmp) > 0:
            cuis_di[pssn] = cuis_li_tmp
            date_di[pssn] = date_li_tmp
    return cuis_di, date_di, dictionary

patients_few = dict(islice(patients.items(), 0, 200))
# cuis, date, dictionary = icd2cui(patients_few, logging_step=50000)
cuis, date, dictionary = icd2cui(patients, logging_step=50000)

dir_apth = '/home/zhichaoyang/mimic3/autoicd_pretrain/data_example'
print("Number of cui in dictionary: {}".format(len(dictionary)), flush=True)
with open(dir_apth + '/dict.txt', 'w') as handle: #TODO
    handle.write("[PAD]"+"\n")
    for i in range(99):
        handle.write("[unused{}]".format(i)+"\n")
    handle.write("[UNK]"+"\n")
    handle.write("[CLS]"+"\n")
    handle.write("[SEP]"+"\n")
    handle.write("[MASK]"+"\n")
    for i in range(99,194):
        handle.write("[unused{}]".format(i)+"\n")
    for k,v in dictionary.items():
        handle.write("{}\n".format(k))
# save data
print("Saving patient data...", flush=True)
f1 = open(dir_apth + '/value.pickle', 'wb') 
f3 = open(dir_apth + '/dates.pickle', 'wb') 
f2 = open(dir_apth + '/key.txt', 'w')
for k,v in cuis.items():
    pickle.dump(v, f1, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(date[k], f3, protocol=pickle.HIGHEST_PROTOCOL)
    f2.write("{}\n".format(k))
f1.close()
f3.close()
f2.close()

print("Done")




