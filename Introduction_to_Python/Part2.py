
import matplotlib.pyplot as plt
import pandas as pd

fd = pd.read_csv(r'F:\University\Spring99\01)Artificial Intelligence\CA0\Project codes\AdmissionPredict.csv')
fd['GRE Score'] = fd['GRE Score'].fillna((fd['GRE Score'].mean()))
fd['TOEFL Score'] = fd['TOEFL Score'].fillna((fd['TOEFL Score'].mean()))
fd['University Rating'] = fd['University Rating'].fillna((fd['University Rating'].mean()))
fd['SOP'] = fd['SOP'].fillna((fd['SOP'].mean()))
fd['LOR '] = fd['LOR '].fillna((fd['LOR '].mean()))
fd['CGPA'] = fd['CGPA'].fillna((fd['CGPA'].mean()))
fd['Research'] = fd['Research'].fillna((fd['Research'].mean()))

x = fd['Research']
y = fd['Chance of Admit']
plt.scatter(x, y, marker='o',color='violet')
plt.xlabel('Research')
plt.ylabel('Chance of Admit')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()
