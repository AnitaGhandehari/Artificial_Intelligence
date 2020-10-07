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

average1 = fd.loc[fd['University Rating'] == 1, 'GRE Score'].mean()
print("Average of GRE Score in University Rating1:  " + str(average1))
average2 = fd.loc[fd['University Rating'] == 2, 'GRE Score'].mean()
print("Average of GRE Score in University Rating2:  " + str(average2))
average3 = fd.loc[fd['University Rating'] == 3, 'GRE Score'].mean()
print("Average of GRE Score in University Rating3:  " + str(average3))
average4 = fd.loc[fd['University Rating'] == 4, 'GRE Score'].mean()
print("Average of GRE Score in University Rating4:  " + str(average4))
average5 = fd.loc[fd['University Rating'] == 5, 'GRE Score'].mean()
print("Average of GRE Score in University Rating5:  " + str(average5))
