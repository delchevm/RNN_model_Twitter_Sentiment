import pandas as pd
import io
import csv
import re

data_folder = "D:/Python/Text Classification Data/Twitter/"
open_file = data_folder + "Twitter_master.csv"
write_folder = data_folder + "Outputs/"
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

with open(open_file, 'r', encoding='utf8', errors='ignore') as file:
    data = csv.reader(file)
    row_nub = 0

    for row in data:
        if int(row[0]) < 3:
            pos_neg = "neg/"
        elif int(row[0]) > 3:
            pos_neg = "pos/"
        else:
            row_nub = row_nub + 1
            continue
        try:

            row[1] = re.sub("@[A-Za-z0-9]+", "", row[1])
            row[1] = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", row[1])

            no_punct = ""
            for char in row[1]:
                if char not in punctuations:
                    no_punct = no_punct + char

            g = open(write_folder + pos_neg + str(row_nub) + '.txt', 'w')
            g.write(no_punct.lower())
            row_nub = row_nub + 1
            g.close()
        except:
            pass


