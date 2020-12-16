import os
import datetime
import glob

os.chdir('/media/sdig/LACROS/cloudnet/data/punta-arenas/processed/limrad94/categorize-py/2019')

#print(os.listdir('*.png'))
print(len(glob.glob("*.png")))
all_pngs = glob.glob("*.png")
file_list = []
for ifile in all_pngs:
    try:
        if datetime.datetime.utcfromtimestamp(os.path.getmtime(ifile)) < datetime.datetime(2020, 9, 26):
            file_list.append(ifile[:8])
            print(f'#"--dt-start {ifile[:8]} --dt-end {ifile[:8]} --h-start 0000 --h-end 2359"')
    except:
        print('something went wrong')

print(sorted(file_list))