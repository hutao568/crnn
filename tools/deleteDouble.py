import json

files=open('test.json','r')
files1=open('test1.json','w')
lines=files.readlines()
for line in lines:
    if line.find('#')==-1:
        files1.write(line)

files.close()
files1.close()
