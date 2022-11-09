from macpath import join
from preprocessor import Preprocessor
import SimpleITk as sitk
import os, glob
from multiprocessing import Process
import pandas as pd
import numpy as np


def multiprocess_pre(lbl_path, dwi_dir, flair_dir, process_num):
    lbl_df = pd.read_csv(lbl_path, sep='')

    for pid in lbl_df['code_n']:
        patient_dwi_dir = os.path.join(dwi_dir, str(pid))
        patient_flair_dir = os.path.join(flair_dir, str(pid))
        if os.path.exists(patient_dwi_dir) and os.path.exists(patient_flair_dir):
            dwi_path = glob.glob(os.path.join(dwi_dir, str(pid), "*.nii.gz"))[0]
            flair_path = glob.glob(os.path.join(flair_dir, str(pid), "*.nii.gz"))[0]

            process_list = []
            for i in range(process_num):
                p = Process(target=preprocess, args=(pid, dwi_path, flair_path))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()


def preprocess(pid, dwi_path, flair_path):
    pre_dwi = Preprocessor(target_spacing=[.9, .9, 6.5])
    pre_flair = Preprocessor(target_spacing=[.9, .9, 6.5])
    sitk_dwi = pre_dwi.run(dwi_path)
    sitk_flair = pre_flair.run(flair_path)
    dwi_arr = sitk.GetArrayFromImage(sitk_dwi)
    flair_arr = sitk.GetArrayFromImage(sitk_flair)
    np.savez(f"../../dataset/{pid}.pnz", dwi=dwi_arr, flair=flair_arr)

process_list = []
data_dir = ""
cpu_count = 8
subjects = list(glob(join(data_dir, "*.nii.gz")))
remainder = len(subjects) % cpu_count
batch = len(subjects) // cpu_count
for i in range(cpu_count):
    if i == (cpu_count - 1):
        this_subjects = subjects[i * batch:]
    else:
        this_subjects = subjects[i*batch : (i+1)*batch]
    p = Process(target=preprocessfun, args=(this_subjects,))
    p.start()
    process_list.append(p)

for _ in process_list:
    p.join()

print("preprocess end!")


if __name__ == '__main__':
    label_path = ""
    dwi_dir = ""
    flair_dir = ""
    multiprocess_pre(label_path, dwi_dir, flair_dir, process_num=40)
