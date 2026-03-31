import bz2

file_path = '../SNAP/wikipedia/enwiki-20080103.main.bz2'

counter = 0
with bz2.open(file_path, mode='rt', encoding='utf-8') as f:
    with open('wikipedia_out.csv', 'w', encoding='utf-8') as out:
        out.write('articleid,userid,datetime\n')
        for line in f:
            counter += 1
            if counter % 1E6 == 0: print(counter / 1E6, "Million lines processed")

            if not line.startswith('REVISION'): continue

            linelist = line.strip().split(' ')
            userid = linelist[-1]
            if userid.startswith('ip:'): continue

            articleid = linelist[1]
            datetime = linelist[4]

            out.write(articleid+','+userid+','+datetime+'\n')