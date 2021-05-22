import os
import string

path = 'D:/Python/TestClassData_model/test/neg/'
files = os.listdir(path)

printable = set(string.printable)

for file in files:
    f = open(path + file, 'r', encoding="utf-8", errors="replace")
    for line in f:
        filter(lambda x: x in printable, line)
        line = ''.join(filter(lambda x: x in printable, line))
    f.close()
    f = open(path + file, 'w')
    f.write(line)
    f.close()
