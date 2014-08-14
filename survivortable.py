# survivortable.py - Extending the "gender-class model" from the tutorials,
# this is an effort to assess the value of adding dimensions to the survival
# table to improve the predictive capability of the model.

import csv as csv
import pandas as pd
import numpy as np

### Training The Model

# The Training Data
#
# 12 features as follows:
#   0.  PassengerId : Integer
#   1.  Survived    : 0 - no, 1 - yes
#   2.  Pclass      : 1, 2, 3
#   3.  Name        : String
#   4.  Sex         : 'male', 'female'
#   5.  Age         : Integer
#   6.  SibSp       : Number of siblings or a spouse, Integer
#   7.  Parch       : Number of parents or children, Integer
#   8.  Ticket      : Ticket serial, String
#   9.  Fare        : Ticket price, Float
#   10. Cabin       : String
#   11. Embarked    : Port of Origin, Character, S = , C = , Q = 

# Load the training data directly into a Pandas data frame.
df = pd.read_csv('data/train.csv', header = 0)

# Add 'Gender' column and map 'Sex' to female = 0 and male = 1.
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Map point of embarkation to 0, 1, 2 -- how to deal with the missing values?
#   1. Assign a random location from the set of three,
#   2. Assign the most common location,
#   3. Assign a random location from a weighted set of the three
# For now I just set missing values to 0 (S).
df['PortOfOrigin'] = df['Embarked'].map( {None: 0, 'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Replace missing ages with median gender-class age -- can we make this
# possibly more accurate by using other features?
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) &
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
# Make a copy of the Age column
df['AgeFill'] = df['Age']
# A column of 0s and 1s indicating which Age records had missing values.
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
# Plug in median gender-class ages for null entries
for i in range(0,2):
    for j in range(0,3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) &(df.Pclass == j+1),
                'AgeFill'] = median_ages[i,j]

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Age*Class'] = df.AgeFill * df.Pclass

# Drop unused columns, and rows with missing values
df = df.drop(['Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.dropna()

# Convert from data frame to array for NumPy work
data = df.values

# We now have 12 features as follows:
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

# Family size needs to be binned. Max family size in the training data is 10.
family_size_ceiling = 10
family_bracket_size = 3
number_of_family_size_brackets = family_size_ceiling / family_bracket_size
# Any family size >= family_size_ceiling is set to family_size_ceiling - 1
data[data[0::,10].astype(np.float) >= family_size_ceiling,10] = family_size_ceiling - 1.0

# Fare prices need to be binned.
fare_ceiling = 40
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size
# Any fare price >= fare_ceiling is set to fare_ceiling - 1
data[data[0::,5].astype(np.float) >= fare_ceiling,5] = fare_ceiling - 1.0

# We know there are three classes but let's pull that from the data.
number_of_classes = len(np.unique(data[0::,2]))

# Initialize our n-dimensional survival table with zeros.
survival_table = np.zeros([2,number_of_classes,number_of_price_brackets,number_of_family_size_brackets],float)

# Populate the survival table with the data from the training set.
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
        for k in xrange(number_of_family_size_brackets):

            # Filter where gender = female and class = i+1 and fare in price bracket j+1
            women_only_stats = data[ (data[0::,6] == 0) \
                                     & (data[0::,2].astype(np.float) == i+1) \
                                     & (data[0:,5].astype(np.float) >= j*fare_bracket_size) \
                                     & (data[0:,5].astype(np.float) < (j+1)*fare_bracket_size) \
                                     & (data[0:,10].astype(np.float) >= k*family_bracket_size) \
                                     & (data[0:,10].astype(np.float) < (k+1)*family_bracket_size), 1]

            # Filter where gender = male and class = i+1 and fare in price bracket j+1
            men_only_stats = data[ (data[0::,6] == 1) \
                                     & (data[0::,2].astype(np.float) == i+1) \
                                     & (data[0:,5].astype(np.float) >= j*fare_bracket_size) \
                                     & (data[0:,5].astype(np.float) < (j+1)*fare_bracket_size) \
                                     & (data[0:,10].astype(np.float) >= k*family_bracket_size) \
                                     & (data[0:,10].astype(np.float) < (k+1)*family_bracket_size), 1]

            survival_table[0,i,j,k] = np.mean(women_only_stats.astype(np.float))  # Female stats
            survival_table[1,i,j,k] = np.mean(men_only_stats.astype(np.float))    # Male stats

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

### Testing
test_data = csv.reader(open('data/test.csv', 'rb'))
header = test_data.next()
predictions_file = csv.writer(open("results/survivortable.csv", "wb"))
predictions_file.writerow(["PassengerId", "Survived"])

# The Test Data
#
# 11 features as follows:
#   0.  PassengerId : Integer
#   1.  Pclass      : 1, 2, 3
#   2.  Name        : String
#   3.  Sex         : 'male', 'female'
#   4.  Age         : Integer
#   5.  SibSp       : Number of siblings or a spouse, Integer
#   6.  Parch       : Number of parents or children, Integer
#   7.  Ticket      : Ticket serial, String
#   8.  Fare        : Ticket price, Float
#   9.  Cabin       : String
#   10. Embarked    : Port of Origin, Character, S = , C = , Q = 

# Now process each row of the test data.
for row in test_data:
    # First thing to do is bin up the price file
    for j in xrange(number_of_price_brackets):
        # If there is no fare then place the price of the ticket according to class
        try:
            row[8] = float(row[8])    # No fare recorded will come up as a string so try to make it a float
        except:                       # If fails then just bin the fare according to the class
            bin_fare = 3 - float(row[1])
            break                     # Break from the loop and move to the next row

        if row[8] > fare_ceiling:     # Otherwise now test to see if it is higher than the fare ceiling we set earlier
            bin_fare = number_of_price_brackets - 1
            break                     # And then break to the next row

        if row[8] >= j*fare_bracket_size\
            and row[8] < (j+1)*fare_bracket_size:     # If passed these tests then loop through
                                                      # each bin until you find the right one
                                                      # append it to the bin_fare
                                                      # and move to the next loop
            bin_fare = j
            break

    # Now bin up the family size
    for k in xrange(number_of_family_size_brackets):
        a = int(row[5]) + int(row[6]) + 1
        if a >= k*family_bracket_size and a < (k+1)*family_bracket_size:
            bin_family_size = k
            break

    if row[3] == 'female':
        predictions_file.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare, bin_family_size ])])
    else:
        predictions_file.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare, bin_family_size ])])

