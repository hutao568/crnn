import json

#train
train=open('data/trainAddPinghu.json')
train_single=open('data/train_sPH.json','w')
train_double=open('data/train_dPH.json','w')
lines=train.readlines()

for line in lines:
    plate=json.loads(line)['label'][0]['data'][0]['class']
    if  line.find('#')==-1:
        train_single.write(line)
    else:
        train_double.write(line)

train.close()
train_single.close()
train_double.close()

#test
test=open('/data/plate/test/test.json')
test_single=open('/data/plate/test/test_s.json','w')
test_double=open('/data/plate/test/test_d.json','w')
lines=test.readlines()

for line in lines:
    plate=json.loads(line)['label'][0]['data'][0]['class']
    if  line.find('#')==-1:
        test_single.write(line)
    else:
        test_double.write(line)

test.close()
test_single.close()
test_double.close()
