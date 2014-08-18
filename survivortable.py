# survivortable.py - Extending the "gender-class model" from the tutorials,
# this is an effort to assess the value of adding dimensions to the survival
# table to improve the predictive capability of the model.

# First attempt: Adding 'FamilySize' as a factor in survival. This did not
# improve my Kaggle score.

# Second attempt: Remove 'FamilySize' and add the 'Age*Class' feature as a
# factor in survival.

import csv as csv
import pandas as pd
import numpy as np

# ## Training The Model

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
df = pd.read_csv('data/train.csv', header=0)

# Map 'Sex' to integer values, female = 0, male = 1.
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Map point of embarkation to 0, 1, 2 -- how to deal with the missing values?
#   1. Assign a random location from the set of three,
#   2. Assign the most common location,
#   3. Assign a random location from a weighted set of the three
# For now I just set missing values to 0 (S).
df['Embarked'] = df['Embarked'].map({None: 0, 'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Replace missing ages with median gender-class age.
median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df[(df['Sex'] == i) &
                               (df['Pclass'] == j + 1)]['Age'].dropna().median()

# Plug in median gender-class ages for null entries
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df['Age'].isnull()) & (df['Sex'] == i) & (df['Pclass'] == j + 1),
               'Age'] = median_ages[i, j]

# Drop unused columns, and rows with missing values
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df = df.dropna()

# Convert from data frame to array for NumPy work
data = df.values

# We now have 11 features as follows:
#   0.  Survived:     0 = no, 1 = yes
#   1.  Pclass:       1 = 1st, 2 = 2nd, 3 = 3rd
#   2.  Sex:          0 = female, 1 = male
#   3.  Age:          Float
#   4.  SibSp:        Int
#   5.  Parch:        Int
#   6.  Fare:         Float
#   7.  Embarked:     0 = , 1 = , 2 = 

# Family size needs to be binned. Max family size in the training data is 10.
family_size_ceiling = 10
family_bracket_size = 3
number_of_family_size_brackets = family_size_ceiling / family_bracket_size
# Any family size >= family_size_ceiling is set to family_size_ceiling - 1
data[data[0::, 10].astype(np.float) >= family_size_ceiling, 10] = family_size_ceiling - 1.0

# Fare prices need to be binned.
fare_ceiling = 40
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size
# Any fare price >= fare_ceiling is set to fare_ceiling - 1
data[data[0::, 5].astype(np.float) >= fare_ceiling, 5] = fare_ceiling - 1.0

# We know there are three classes but let's pull that from the data.
number_of_classes = len(np.unique(data[0::, 2]))

# Initialize our n-dimensional survival table with zeros.
survival_table = np.zeros([2, number_of_classes, number_of_price_brackets, number_of_family_size_brackets], float)

# Populate the survival table with the data from the training set.
for i in range(0, 2):
    for j in xrange(number_of_classes):
        for k in xrange(number_of_price_brackets):
            for l in xrange(number_of_family_size_brackets):

                stats = data[(data[0::, 6] == i) \
                             & (data[0::, 2].astype(np.float) == j + 1) \
                             & (data[0:, 5].astype(np.float) >= k * fare_bracket_size) \
                             & (data[0:, 5].astype(np.float) < (k + 1) * fare_bracket_size) \
                             & (data[0:, 10].astype(np.float) >= l * family_bracket_size) \
                             & (data[0:, 10].astype(np.float) < (l + 1) * family_bracket_size), 1]

                if not stats:
                    survival_table[i, j, k, l] = 0
                else:
                    survival_table[i, j, k, l] = np.mean(stats.astype(np.float))

survival_table[survival_table != survival_table] = 0

# Now I have my proportion of survivors, simply round them such that if <0.5
# I predict they don't survive, and if >= 0.5 they do
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

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
            row[8] = float(row[8])  # No fare recorded will come up as a string so try to make it a float
        except:  # If fails then just bin the fare according to the class
            bin_fare = 3 - float(row[1])
            break  # Break from the loop and move to the next row

        if row[8] > fare_ceiling:  # Otherwise now test to see if it is higher than the fare ceiling we set earlier
            bin_fare = number_of_price_brackets - 1
            break  # And then break to the next row

        if j * fare_bracket_size <= row[8] < (j + 1) * fare_bracket_size:  # If passed these tests then loop through
            # each bin until you find the right one
            # append it to the bin_fare
            # and move to the next loop
            bin_fare = j
            break

    # Now bin up the family size
    for k in xrange(number_of_family_size_brackets):
        a = int(row[5]) + int(row[6]) + 1
        if k * family_bracket_size <= a < (k + 1) * family_bracket_size:
            bin_family_size = k
            break

    if row[3] == 'female':
        predictions_file.writerow([row[0], "%d" % int(survival_table[0, float(row[1]) - 1, bin_fare, bin_family_size])])
    else:
        predictions_file.writerow([row[0], "%d" % int(survival_table[1, float(row[1]) - 1, bin_fare, bin_family_size])])

