import json

files1=open('/data/plate/train/train.json','a')
files2=open('/data/plate/train/pinghu1.json')

lines=files2.readlines()

for line in lines:
    files1.write(line)

files1.close()
files2.close()