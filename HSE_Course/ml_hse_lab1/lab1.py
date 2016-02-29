# -*- coding: utf-8 -*-

import pandas
from collections import Counter

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# print(data['Sex'].value_counts())

survived = data['Survived'].value_counts()
# print(survived)
# print(survived[1])

all_count = survived[0] + survived[1]

percent_dead = 100.0*survived[0]/all_count
percent_survived = 100.0*survived[1]/all_count

# print(percent_survived)
# print(percent_dead)
# print(percent_dead+percent_survived)

pclass = data['Pclass'].value_counts()
# print(100.0*pclass[1]/(pclass[1]+pclass[2]+pclass[3]))


# print(data['SibSp'].corr(data['Parch'], method='pearson'))

women = data.loc(data['Sex'] == 'female')

female_names = data[data['Sex'] == 'female']['Name']

count_arr = []

for name in female_names:
    count_arr.append(name.split(', ')[1])

# print(count_arr[0])
print(Counter(count_arr).most_common(1))
