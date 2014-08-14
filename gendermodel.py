# gendermodel.py - A very naive model for predicting Titanic survivors. A
# trimmed version of the tutorial code from the Titanic Kaggle competition.

import csv as csv
import numpy as np

### Train

# Read the training data into a list and convert to a NumPy array.
training_data = csv.reader(open('data/train.csv', 'rb'))
header = training_data.next()    # Lop off the header.
data=[]
for row in training_data:
    data.append(row[0:])
data = np.array(data)

# There are 891 records of 12 features each. At this point all the feature
# values are strings. All the values of a NumPy array must be of the same type.
# Since the data is of mixed type, strings are required as the only means to
# represent all the different features. We will need to convert numeric types
# on the fly. Pandas fixes this by offering data frames.

# Extract all female and male records into separate variables. We can filter
# inside the array indices much as we would in R.
females = data[data[0::,4]=="female", 1].astype(np.float)
males = data[data[0::,4]=="male", 1].astype(np.float)
# Calculate pere-gender survivor ratios.
female_survival_ratio = np.sum(females) / np.size(females)
male_survival_ratio = np.sum(males) / np.size(males)
print 'Proportion of women who survived is %s' % female_survival_ratio
print 'Proportion of men who survived is %s' % male_survival_ratio

### Test

# So we know a majority fo females survived while a majority of males perished.
# Based on this information, our simple survivor model is: if the passenger is
# female, they survive, if male, they do not survive.

# Open the test data set and lop off the header. Open a new file for recording
# our results.
test_data = csv.reader(open('data/test.csv', 'rb'))
header = test_data.next()
results = csv.writer(open("results/gendermodel.csv", "wb"))
results.writerow(["PassengerId", "Survived"])

# Loop through all the records in the test data and write the model results to
# our output file.
for row in test_data:
    if row[3] == 'female':
        results.writerow([row[0], "1"])
    else:
        results.writerow([row[0], "0"])
#test_data.close()
#results.close()
