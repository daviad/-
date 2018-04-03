
#!/usr/bin/python3
# coding=utf-8

import os, fnmatch,  sys
# 打开文件
fd = os.open("../train/data.txt",os.O_RDWR|os.O_CREAT)

# ella 0
# selina 1
for root, dirs, files in os.walk("../face/ella", topdown=False):
    for name in files:
        text = (root +"/"+ name + " " + "0" + "\n").encode();
        print(text)
        os.write(fd,text); 

for root, dirs, files in os.walk("../face/selina", topdown=False):
    for name in files:
        text = (root  +"/"+  name + " " + "1" + "\n").encode();
        print(text)
        os.write(fd,text); 

os.close(fd)


