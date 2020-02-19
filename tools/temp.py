import json

files1=open('data/train.json','a')
files2=open('data/pinghu1.json')

lines=files2.readlines()

for line in lines:
    files1.write(line)

files1.close()
files2.close()
