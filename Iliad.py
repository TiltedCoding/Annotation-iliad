import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score as kappa
from itertools import combinations as com

files = os.listdir() # Get the files from the path
filenames = [f for f in files if f[-4:] == 'xlsx'] # Choose only excel files.
print(filenames)
os.chdir(path) # Change Directory so that it can read the excel files.
coders = pd.DataFrame() # Initiate the Data Frame
for f in filenames:
    df = pd.read_excel(f)
    df['ID'] = int(f[1:-5]) # Create the ID column with the file's names as the annotators
    if int(f[-6]) < 5:
        df['Group'] = 'G2'
    else:
        df['Group'] = 'G1'
    coders = coders.append(df) # Create the Data Frame
    
print('Sentiment unique data: ', coders.Sentiment.unique(), '\nSubjectivity unique data: ', coders.Subjectivity.unique(), '\nPrimary Emotion unique data:\n', coders['Primary Emotion'].unique())
# Strip the column of unwanted whitespace. Zero's become NaNs but later we will turn all NaNs back to zeros.
coders.Sentiment = coders.Sentiment.str.strip()
# Replace the sentiments given with values  -1 0 1. The symbol _ is taken into account as a minus so it is converted to -1
coders.Sentiment.replace(['m', '+', '-', '_'], [1, 1, -1, -1], inplace=True)
# Check if there are NaNs in Id_verse before making it an index.
coders['Id_verse'].isnull().sum()
# Delete all the rows that contain missing values as Text.
coders = coders.loc[coders['Text'].notnull()]
# Delete all the rows that contain missing vallues as Id_verse since there is no data for these rows except for Text.
coders = coders.loc[coders['Id_verse'].notnull()]
# Drop all the unwanted columns.
coders.drop(['Seocondary Emotions', 'Emotion Primary', 'Emotion Secondary', 'Secondary Emotion'], axis=1, inplace=True)
# Check if there are missing values in Subjectivity and Sentiment Columns.
coders.isna().sum()
# Fill all the Sentiment NaNs with 0. Furthermore, assuming that NaNs are equivalent to a zero for the annotator (We assume that the annotator left the cell blank instead of putting zero) we replace NaNs with 0s for Subjectivity as well. Finally, replace NaN's with neutral emotion in Primary Emotions.
coders.Subjectivity.fillna(0, inplace=True)
coders.Sentiment.fillna(0, inplace=True)
coders['Primary Emotion'].fillna('neutral', inplace=True)
# Check that the missing values problem is resolved in each Column.
coders.isna().sum()
coders.sample(10)
# Check the unique ID's and their type.
coders.ID.unique()
# Convert the whole ID
coders.ID = coders.ID.astype('str')
# Set Group, ID and Id_verse as indexes
coders.set_index(['Group', 'ID', 'Id_verse'], inplace=True)
coders.sample(10)
# Confirm that the indices were changed correctly
coders.index.names
# Make lists of each available combination of annotators for each group using the combinations tool from the library "itertools".
# Reminder that we imported combinations as com, hence the com after the list.
G1comb = list(com(coders.loc['G1'].index.get_level_values(0).unique(), 2))
G2comb = list(com(coders.loc['G2'].index.get_level_values(0).unique(), 2))
# Check if the combinations are correct.
print('G1 Combinations:')
for i in range(len(G1comb)):
    print(G1comb[i][0], 'with' ,G1comb[i][1])
print('G2 Combinations:')
for i in range(len(G2comb)):
    print(G2comb[i][0], 'with' , G2comb[i][1])
# Initialize lists for each cohen kappa.
k11 = []
k12 = []
k21 = []
k22 = []
#Use for loops in order to run through the combinations of the annotators and get a kappa value for each combination.
for i in range (0, len(G1comb)):
    k11.append(kappa(coders.loc['G1', G1comb[i][0]].Sentiment, coders.loc['G1', G1comb[i][1]].Sentiment))
    k12.append(kappa(coders.loc['G1', G1comb[i][0]].Subjectivity, coders.loc['G1', G1comb[i][1]].Subjectivity))

for j in range (0, len(G2comb)):
    k21.append(kappa(coders.loc['G2', G2comb[j][0]].Sentiment, coders.loc['G2', G2comb[j][1]].Sentiment))
    k22.append(kappa(coders.loc['G2', G2comb[j][0]].Subjectivity, coders.loc['G2', G2comb[j][1]].Subjectivity))

k11 = np.array(k11)
k12 = np.array(k12)
k21 = np.array(k21)
k22 = np.array(k22)
# Compute the mean value for each kappa in the list of kappas and print it to compare the values.
print(f'G1 Sentiment Cohen kappa: {np.mean(k11):.4f}\nG1 Subjectivity Cohen kappa: {np.mean(k12):.4f}\nG2 Sentiment Cohen kappa: {np.mean(k21):.4f}\nG2 Subjectivity Cohen kappa: {np.mean(k22):.4f}')

# Again, initialize lists for each group for sentiment and subjectivity. These lists will contain the agreement for each pair in the combinations list.
G1Sen = []
G1Sub = []
G2Sen = []
G2Sub = []
# Use for loops in order to go throught the lists of combinations. For each group/sentiment/subjectivity pair, a dataframe is computed. Then using percentage agreement with logical not xor as in the data_annotations pdf, we append each agreement percentage of each pair of the combinations list into the lists initialized above.
for i in range (0, len(G1comb)):
    # First make the data frame consisting of each pair's choices. 
    annots11 = pd.DataFrame(pd.concat([coders.loc['G1', G1comb[i][0]].Sentiment, coders.loc['G1', G1comb[i][1]].Sentiment], axis=1))
    # Change column names so that not both columns of the Data Frame are called by the same name (Sentiment).
    annots11.columns = ['Sentiment11', 'Sentiment12']
    # Use not logical XOR to find the agreements between each row. After that, the matrix is appended to the list.
    G1Sen.append(annots11.apply(lambda r: not np.logical_xor(r.Sentiment11, r.Sentiment12), axis=1))
    
    annots12 = pd.DataFrame(pd.concat([coders.loc['G1', G1comb[i][0]].Subjectivity, coders.loc['G1', G1comb[i][1]].Subjectivity], axis=1))
    annots12.columns = ['Subjectivity11', 'Subjectivity12']
    G1Sub.append(annots12.apply(lambda r: not np.logical_xor(r.Subjectivity11, r.Subjectivity12), axis=1))

for j in range (0, len(G2comb)):
    annots21 = pd.DataFrame(pd.concat([coders.loc['G2', G2comb[j][0]].Sentiment, coders.loc['G2', G2comb[j][1]].Sentiment], axis=1))
    annots21.columns = ['Sentiment21', 'Sentiment22']
    G2Sen.append(annots21.apply(lambda r: not np.logical_xor(r.Sentiment21, r.Sentiment22), axis=1))
    
    annots22 = pd.DataFrame(pd.concat([coders.loc['G2', G2comb[j][0]].Subjectivity, coders.loc['G2', G2comb[j][1]].Subjectivity], axis=1))
    annots22.columns = ['Subjectivity21', 'Subjectivity22']
    G2Sub.append(annots22.apply(lambda r: not np.logical_xor(r.Subjectivity21, r.Subjectivity22), axis=1))
# Compute mean agreement percentage for each Group.
print(f'Percentage Agreement of G1 for Sentiment: {np.mean(G1Sen)*100:.2f}%\nPercentage Agreement of G1 for Subjecctivity: {np.mean(G1Sub)*100:.2f}%\nPercentage Agreement of G2 for Sentiment: {np.mean(G2Sen)*100:.2f}%\nPercentage Agreement of G2 for Subjectivity: {np.mean(G2Sub)*100:.2f}%')
