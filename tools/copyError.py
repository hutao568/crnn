import shutil
import json
files=open('error1.txt','r')
files1=open('train_s.json','r')
lines=files.readlines()
labels=files1.readlines()

for i in range(0,len(lines),3):
    index=int(lines[i])
    plate=json.loads(labels[index])['label'][0]['data'][0]['class']
    src=json.loads(labels[index])['url'].replace('qiniu:///supredata-internal-ocr/', '/data/', 1)
    dst='/data/plate/error/'+lines[i+1][:-1]+'.png'
    shutil.copyfile( src, dst)

files.close()
files1.close()
