# myfirstforest.py - A model that implements a random forest. Will it do any
# better than the previous and seemingly more naive models?

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

### Training Data
training_df = pd.read_csv('data/train.csv', header = 0)

## Clean

# Map gender to 0 (female) and 1 (male)
training_df['Gender'] = training_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Map point of embarkation to 0, 1, 2, 3 -- How should we deal with missing
# values?
training_df['PortOfOrigin'] = training_df['Embarked'].map( {None: 0, 'S': 1, 'C': 2, 'Q': 3} ).astype(int)

# Make a copy of the Age column
training_df['AgeFill'] = training_df['Age']
# Calculate median per-class ages
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = training_df[(training_df['Gender'] == i) &
            (training_df['Pclass'] == j+1)]['Age'].dropna().median()
# Plug in median gender-class ages for null entries
for i in range(0,2):
    for j in range(0,3):
        training_df.loc[ (training_df.Age.isnull()) & (training_df.Gender == i) &(training_df.Pclass == j+1),
                'AgeFill'] = median_ages[i,j]
# A column of 0s and 1s indicating which records had missing Age values.
training_df['AgeIsNull'] = pd.isnull(training_df.Age).astype(int)

training_df['FamilySize'] = training_df['SibSp'] + training_df['Parch']
training_df['Age*Class'] = training_df.AgeFill * training_df.Pclass

# Drop unused columns and rows with missing values
training_df = training_df.drop(['PassengerId', 'Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
training_df = training_df.dropna()

# All the missing Fares -> assume median of their respective class
if len(training_df.Fare[ training_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):
        median_fare[f] = training_df[ training_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):
        training_df.loc[ (training_df.Fare.isnull()) & (training_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Convert training data back to a NumPy array
training_data = training_df.values

# We have 11 features as follows:
#   1.  Survived:     0 =no, 1 = yes
#   2.  Pclass:       1 = 1st, 2 = 2nd, 3 = 3rd
#   3.  SibSp:        Int
#   4.  Parch:        Int
#   5.  Fare:         Float
#   6.  Gender:       0 = female, 1 = male
#   7.  PortOfOrigin: 0 = missing, 1 = , 2 = , 3 = 
#   8.  AgeFill:      Float
#   9.  AgeIsNull:    0 | 1
#   10. FamilySize:   Int (= SibSp + Parch)
#   11. Age*Class:    Float (= AgeFill * Pclass)

### Test Data
test_df = pd.read_csv('data/test.csv', header=0)

# Map gender to 0 (female) and 1 (male)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Map point of embarkation to 0, 1, 2
test_df['PortOfOrigin'] = test_df['Embarked'].map( {None: 0, 'S': 1, 'C': 2, 'Q': 3} ).astype(int)

# Make a copy of the Age column
test_df['AgeFill'] = test_df['Age']
# Calculate median per-class ages.
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = test_df[(test_df['Gender'] == i) &
                (test_df['Pclass'] == j+1)]['Age'].dropna().median()
# Plug in median gender-class ages for null entries
for i in range(0,2):
    for j in range(0,3):
        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) &(test_df.Pclass == j+1),
                'AgeFill'] = median_ages[i,j]
# A column of 0s and 1s indicating which records had missing Age values.
test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values

# Drop unused columns and rows with missing values
test_df = test_df.drop(['PassengerId', 'Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.dropna()

test_data = test_df.values

# We have 11 features as follows:
#   0.  Survived:     0 =no, 1 = yes
#   1.  Pclass:       1 = 1st, 2 = 2nd, 3 = 3rd
#   2.  SibSp:        Int
#   3.  Parch:        Int
#   4.  Fare:         Float
#   5.  Gender:       0 = female, 1 = male
#   6.  PortOfOrigin: 0 = missing, 1 = , 2 = , 3 = 
#   7.  AgeFill:      Float
#   8.  AgeIsNull:    0 | 1
#   9.  FamilySize:   Int (= SibSp + Parch)
#   10. Age*Class:    Float (= AgeFill * Pclass)

### Run The Model
print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
# Second arg = "the answer", first arg = the training data
forest = forest.fit( training_data[0::,1::], training_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("results/myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
