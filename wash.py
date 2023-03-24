import csv

lines = []


ycnt = 0
lcnt = 0
txtf = open('data/data_SQLv3.csv', 'w', encoding='utf8')
with open('data/SQLiV3.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['Sentence'],row['Label'])
        if str(row['Label']) == '0' or str(row['Label']) == '1':
            ycnt = ycnt + 1
            txtf.write(row['Sentence']+'\t'+row['Label']+'\n')
        else:
            print(row)
            lcnt = lcnt + 1

csvfile.close()

txtf.close()

print(ycnt,lcnt)

