import wfdb
import wfdb.processing

unovis_data = {}

for i in range (51, 201):
    if i == 194:
        continue
    str_i = str(i)
    unovis_data[i] = wfdb.rdrecord(f"/media/medit-student/Volume/alperen/studydata/UnoViS_BigD_{str_i}/UnoViS_BigD_{str_i}")

print(unovis_data.keys())

