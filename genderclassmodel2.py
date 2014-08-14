# genderclassmodel2.py - A slightly less naive model for predicting Titanic
# survivors. We incorporate the ticket fare price and ticket / cabin class in
# the model. This version uses a Pandas data frame instead of a raw NumPy
# array, and also adds a number of potentially useful features to the training
# data.

import csv as csv
import pandas as pd
import numpy as np

### Train

## Load the training data directly into a Pandas data frame.
df = pd.read_csv('data/train.csv', header = 0)

## Clean
# Map gender to 0 (female) and 1 (male)
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# Map point of embarkation to 0, 1, 2
df['PortOfOrigin'] = df['Embarked'].map( {None: 0, 'S': 1, 'C': 2, 'Q': 3} ).astype(int)
# Replace missing ages with median passenger class age
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) &
                              (df['Pclass'] == j+1)]['Age'].dropna().median()

# Make a copy of the Age column
df['AgeFill'] = df['Age']
# Plug in median gender-class ages for null entries
for i in range(0,2):
    for j in range(0,3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) &(df.Pclass == j+1),
                'AgeFill'] = median_ages[i,j]

# A column of 0s and 1s indicating which records had missing Age values.
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

# Drop unused columns and rows with missing values
df = df.drop(['Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.dropna()

#print df.describe()
#print df.head(5)

# Convert from data frame to array for NumPy work
data = df.values

# We have 12 features as follows:
#   0.  PassengerId:  Int
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

# Fare prices range from 0 (zero = unknown?) to 4.0125 (lowest fare paid) to
# 512.3292 (highest fare paid). In the tutorial online we created four bins
# where the top bin maxed out at 39. Any price paid that was higher than 39 was
# simply set to 39. It would be useful to have more insight into fare price vs.
# ticket class. We can play with the fare_ceiling variable to see what effect
# that has on the results.

# Any fare price over fare_ceiling is set to fare_ceiling -1
fare_ceiling = 40
data[data[0::,5].astype(np.float) >= fare_ceiling,5] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

number_of_classes = len(np.unique(data[0::,2]))

# This reference matrix will show the proportion of survivors as a sorted table
# of gender, class and ticket fare. First initialize it with all zeros.
survival_table = np.zeros([2,number_of_classes,number_of_price_brackets],float)

# I can now find the stats of all the women and men on board
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):

        # Filter where gender = female and class = i+1 and fare in price bracket j+1
        women_only_stats = data[ (data[0::,6] == 0) \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0:,5].astype(np.float) >= j*fare_bracket_size) \
                                 & (data[0:,5].astype(np.float) < (j+1)*fare_bracket_size), 1]

        # Filter where gender = male and class = i+1 and fare in price bracket j+1
        men_only_stats = data[ (data[0::,6] == 1) \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0:,5].astype(np.float) >= j*fare_bracket_size) \
                                 & (data[0:,5].astype(np.float) < (j+1)*fare_bracket_size), 1]

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))  # Female stats
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))    # Male stats

# Since in python if it tries to find the mean of an array with nothing in it
# (such that the denominator is 0), then it returns nan, we can convert these
# to 0 by just saying where does the array not equal the array, and set these
# to 0.
survival_table[ survival_table != survival_table ] = 0.

# Now I have my proportion of survivors, simply round them such that if <0.5
# I predict they dont surivive, and if >= 0.5 they do
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1

#print survival_table

# Now I have my indicator I can read in the test file and write out
# if a women then survived(1) if a man then did not survived (0)
# First read in test
test_file = open('data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Also open the a new file so I can write to it. 
predictions_file = open("results/genderclassmodel.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])

# First thing to do is bin up the price file
for row in test_file_object:
    for j in xrange(number_of_price_brackets):
        # If there is no fare then place the price of the ticket according to class
        try:
            row[8] = float(row[8])    # No fare recorded will come up as a string so
                                      # try to make it a float
        except:                       # If fails then just bin the fare according to the class
            bin_fare = 3 - float(row[1])
            break                     # Break from the loop and move to the next row
        if row[8] > fare_ceiling:     # Otherwise now test to see if it is higher
                                      # than the fare ceiling we set earlier
            bin_fare = number_of_price_brackets - 1
            break                     # And then break to the next row

        if row[8] >= j*fare_bracket_size\
            and row[8] < (j+1)*fare_bracket_size:     # If passed these tests then loop through
                                                      # each bin until you find the right one
                                                      # append it to the bin_fare
                                                      # and move to the next loop
            bin_fare = j
            break
        # Now I have the binned fare, passenger class, and whether female or male, we can
        # just cross ref their details with our survival table
    if row[3] == 'female':
        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare ])])
    else:
        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare])])

# Close out the files
test_file.close()
predictions_file.close()
