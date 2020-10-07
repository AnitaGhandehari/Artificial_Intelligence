import pandas as pd

fd = pd.read_csv(r'F:\University\Spring99\01)Artificial Intelligence\CA0\Project codes\AdmissionPredict.csv')

#print(fd.head(400))
#print(fd.describe(include='all'))
#print(fd.info())
#print(fd.isna())
#print(fd.apply(lambda x: x.fillna(x.mean()), axis=0))
fd['GRE Score'] = fd['GRE Score'].fillna((fd['GRE Score'].mean()))
fd['TOEFL Score'] = fd['TOEFL Score'].fillna((fd['TOEFL Score'].mean()))
fd['University Rating'] = fd['University Rating'].fillna((fd['University Rating'].mean()))
fd['SOP'] = fd['SOP'].fillna((fd['SOP'].mean()))
fd['LOR '] = fd['LOR '].fillna((fd['LOR '].mean()))
fd['CGPA'] = fd['CGPA'].fillna((fd['CGPA'].mean()))
fd['Research'] = fd['Research'].fillna((fd['Research'].mean()))
#print(fd.isna())
print(fd.head(400))
