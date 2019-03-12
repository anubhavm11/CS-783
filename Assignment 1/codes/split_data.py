import os
import shutil

source = '../dataset/train/'
dest = '../dataset/valid/'


files = os.listdir(source)

for f in files:
	cnt = 0
	os.mkdir(dest+f)
	for img in os.listdir(source + f):
		cnt+=1
	cnt2 = 0
	for img in os.listdir(source + f):
		if(cnt2<(0.1*cnt)):
			shutil.move(source+f+'/'+img, dest+f+'/'+img)
		cnt2+=1