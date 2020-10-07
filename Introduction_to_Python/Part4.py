import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fd = pd.read_csv(r'F:\University\Spring99\01)Artificial Intelligence\CA0\Project codes\AdmissionPredict.csv')
fd['GRE Score'] = fd['GRE Score'].fillna((fd['GRE Score'].mean()))
fd['TOEFL Score'] = fd['TOEFL Score'].fillna((fd['TOEFL Score'].mean()))
fd['University Rating'] = fd['University Rating'].fillna((fd['University Rating'].mean()))
fd['SOP'] = fd['SOP'].fillna((fd['SOP'].mean()))
fd['LOR '] = fd['LOR '].fillna((fd['LOR '].mean()))
fd['CGPA'] = fd['CGPA'].fillna((fd['CGPA'].mean()))
fd['Research'] = fd['Research'].fillna((fd['Research'].mean()))

new = fd[['CGPA', 'Chance of Admit']].copy()
j = 0

h = np.ones((new.shape[0], 1))
for i in range(0, new.shape[0]):
    h[i] = -1.25 + 0.23 * new['CGPA'][i]

print(new['Chance of Admit'])
for i in range(0, new.shape[0]):
    if str(new['Chance of Admit'][i]) != 'nan':
        j = j + (h[i] - new['Chance of Admit'][i])
j = j / 768
print("J(Teta_1,Teta_0): " + str(j))

x = new['CGPA']
y = new['Chance of Admit']
plt.scatter(x, y, marker='o',color='purple')
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.plot(x, h, color='red')
#plt.show()

for i in range(0, new.shape[0]):
    if str(new['Chance of Admit'][i]) == 'nan':
        new['Chance of Admit'][i]= -1.25 + 0.23 * new['CGPA'][i]

