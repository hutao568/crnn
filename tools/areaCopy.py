import json
files=open('/data/plate/train/train.json','r')
files1=open('/data/plate/train/train_b.json','w')
lines=files.readlines()

dic='津渝冀晋蒙辽吉黑京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学挂'

for line in lines:
    plate=json.loads(line)['label'][0]['data'][0]['class']
    if dic.find(plate[0]):
        for i in range(10):
            files1.write(line)
    
files.close()
files1.close()