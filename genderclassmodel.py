# genderclassmodel.py - A slightly less naive model for predicting Titanic
# survivors. We incorporate the ticket fare price and ticket / cabin class in
# the model. A trimmed version of the tutorial code from the Titanic Kaggle
# competition.      

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

# Fare prices range from 0 (zero = unknown?) to 4.0125 (lowest fare paid) to
# 512.3292 (highest fare paid). In the tutorial online we created four bins
# where the top bin maxed out at 39. Any price paid that was higher than 39 was
# simply set to 39. It would be useful to have more insight into fare price vs.
# ticket class. We can play with the fare_ceiling variable to see what effect
# that has on the results.

# As a test I changed the fare_ceiling from 40 to 100. There was no change in
# the resulting score on Kaggle.

# And fare price over fare_ceiling is set to fare_ceiling -1
fare_ceiling = 100
data[data[0::,9].astype(np.float) >= fare_ceiling,9] = fare_ceiling - 1.0

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
        women_only_stats = data[ (data[0::,4] == "female") \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                                 & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1]

        # Filter where gender = male and class = i+1 and fare in price bracket j+1
        men_only_stats = data[ (data[0::,4] != "female") \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                                 & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1]

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
