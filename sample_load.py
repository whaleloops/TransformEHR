import pickle

dir_path = "./sample_data"
do_date = True


f1 = open(dir_path+ '/value.pickle', 'rb') 
f3 = open(dir_path+ '/dates.pickle', 'rb') 
f2 = open(dir_path+ '/key.txt', 'r')
keys = f2.readlines()

patients = {}
for key in keys:
    patient_idd = key.strip()
    each_visit = pickle.load(f1)
    f1obj = []
    for (cuis, ext_cuis, strs) in each_visit:
        f1obj.append((cuis, [], []))
    f3obj = pickle.load(f3)
    assert len(f1obj) == len(f3obj)
    if do_date:
        if patient_idd in patients:
            patients[patient_idd] += list(zip(f1obj, f3obj))
        else:
            patients[patient_idd] = list(zip(f1obj, f3obj))
    else:
        if patient_idd in patients:
            patients[patient_idd] += f1obj
        else:
            patients[patient_idd] = f1obj

print("number of patients in the sample dataset")
print(len(patients))
print(patients)
print()