from sklearn import neighbors,svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from seaborn import axes_style
import numpy as np
import math
import copy
import sklearn.preprocessing as preprocessing

columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
           "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
TRAIN_DATA = pd.read_csv('C:\\Users\\raushan jha\\Desktop\\IDS_Project\\adult-train.csv', names=columns, sep=' *, *', na_values='?', engine='python')
TEST_DATA = pd.read_csv('C:\\Users\\raushan jha\\Desktop\\IDS_Project\\adult-test.csv', names=columns, sep=' *, *', skiprows=1, na_values='?', engine='python')


# ******************************** TRAIN_DATA **********************************

# *********************************** EDA **************************************

print(TRAIN_DATA.head())
print(TRAIN_DATA.info())

numerical_attributes = TRAIN_DATA.select_dtypes(include=['int64'])
print(numerical_attributes.columns)


fig = plt.figure(figsize=(20,20))
cols = 3
rows = math.ceil(float(TRAIN_DATA.shape[1]) / cols)
for i, column in enumerate(TRAIN_DATA.columns[:3]):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.set_title(column)
    if TRAIN_DATA.dtypes[column] == np.object:
        TRAIN_DATA[column].value_counts().plot(kind="bar", axes=ax)
    else:
        TRAIN_DATA[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

print(TRAIN_DATA.describe())

categorical_attributes = TRAIN_DATA.select_dtypes(include=['object'])
sns.set(style = 'darkgrid')
with axes_style({'grid.color': "red"}):
    sns.countplot(y='workClass', hue='income', data = categorical_attributes)
plt.show()

with axes_style({'grid.color': "red"}):
    sns.countplot(y='occupation', hue='income', data = categorical_attributes)
plt.show()

with axes_style({'grid.color': "red"}):
    sns.countplot(y='sex', hue='income', data = categorical_attributes)
plt.show()

with axes_style({'grid.color': "red"}):
    sns.countplot(y='education', hue='income', data = categorical_attributes)
plt.show()

numerical_attributes = TRAIN_DATA.select_dtypes(include=['int64'])
sns.heatmap(numerical_attributes.corr(), square=True)
plt.show()

#exit()


# ******************************************************************************

nulls = TRAIN_DATA.isnull().sum()
# print(nulls[nulls > 0])

TRAIN_DATA['workClass'].fillna(TRAIN_DATA['workClass'].value_counts().index[0], inplace=True)
TRAIN_DATA['occupation'].fillna(TRAIN_DATA['occupation'].value_counts().index[0], inplace=True)
TRAIN_DATA['native-country'].fillna(TRAIN_DATA['native-country'].value_counts().index[0], inplace=True)

# print("Missing Values have been Handled")
nulls = TRAIN_DATA.isnull().sum()
# print(nulls[nulls > 0])

# print(TRAIN_DATA.info())

# print("\nTrain Data Size: (rows, cols): ",TRAIN_DATA.shape)
# print("\n")

df = TRAIN_DATA[TRAIN_DATA.duplicated(keep=False)]
# print(df['age'].value_counts())
# print(df)

TRAIN_DATA = TRAIN_DATA.drop_duplicates(keep='first')

# print("\nTrain Data Size: (rows, cols): ",TRAIN_DATA.shape)
# print("\n")

dictmap_sex = {'Male': 0, 'Female': 1}
TRAIN_DATA['sex'] = TRAIN_DATA['sex'].map(dictmap_sex)
# print(TRAIN_DATA['sex'].head(10))

dictmap_edu = {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8,
               'Prof-school': 9, 'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college': 12, 'Bachelors': 13, 'Masters': 14, 'Doctorate': 15}
TRAIN_DATA['education'] = TRAIN_DATA['education'].map(dictmap_edu)

#  Not-working = 0, Gov = 7, Self-emp = 9, Private = 10
dictmap_wrkCl = {'Private': 'Private', 'Self-emp-not-inc': 'Self-emp', 'Self-emp-inc': 'Self-emp', 'Local-gov': 'Gov', 'State-gov': 'Gov',
                 'Federal-gov': 'Gov', 'Without-pay': 'Not-working', 'Never-worked': 'Not-working'}
TRAIN_DATA['workClass'] = TRAIN_DATA['workClass'].map(dictmap_wrkCl)

dictmap_Occ = {'Adm-clerical': 'Admin', 'Armed-Forces': 'Military', 'Craft-repair': 'Blue-Collar', 'Exec-managerial': 'White-Collar',
               'Farming-fishing': 'Blue-Collar', 'Handlers-cleaners': 'Blue-Collar', 'Machine-op-inspct': 'Blue-Collar', 'Other-service':
               'Service', 'Priv-house-serv': 'Service', 'Prof-specialty':'Professional', 'Protective-serv': 'Other-Occupations',
               'Sales': 'Sales', 'Tech-support': 'Other-Occupations', 'Transport-moving': 'Blue-Collar'}
TRAIN_DATA['occupation'] = TRAIN_DATA['occupation'].map(dictmap_Occ)

dictmap_MarSt = {'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married', 'Married-civ-spouse': 'Married',
                 'Married-spouse-absent': 'Not-Married', 'Separated': 'Not-Married', 'Divorced': 'Not-Married', 'Widowed': 'Widowed'}
TRAIN_DATA['marital-status'] = TRAIN_DATA['marital-status'].map(dictmap_MarSt)

TRAIN_DATA['marital_relation'] = TRAIN_DATA['marital-status'] + ' ' + TRAIN_DATA['relationship']
TRAIN_DATA = pd.get_dummies(TRAIN_DATA,columns=['marital_relation'])

dictmap_natCntry = {'United-States':27776, 'Cuba':2621, 'Jamaica':2156, 'India':342, 'Mexico':5715, 'South':3447,
                    'Puerto-Rico':10876, 'Honduras':618, 'England':19709, 'Canada':19859, 'Germany':27087 , 'Iran':1202,
                    'Philippines':939, 'Italy':19273, 'Poland':2874, 'Columbia':2218, 'Cambodia':270, 'Thailand':2490,
                    'Ecuador':2028, 'Laos':325, 'Taiwan':3225, 'Haiti':282, 'Portugal':9978, 'Dominican-Republic':1891,
                    'El-Salvador':1384, 'France':23496, 'Guatemala':1276, 'China':473, 'Japan':39268, 'Yugoslavia':4250,
                    'Peru':1900, 'Outlying-US(Guam-USVI-etc)': 34954, 'Scotland':31367, 'Trinadad&Tobago':3956,
                    'Greece':11091, 'Nicaragua':854, 'Vietnam':220, 'Hong':22502,'Ireland':12430, 'Hungary':4172,
                    'Holand-Netherlands':24331}

gdpvals=[]
cLossvals = []
cGainvals = []
for v in dictmap_natCntry.values():
    gdpvals.append(v)

lowerquartile = int (np.percentile(gdpvals, 25))
median = int (np.percentile(gdpvals, 50))
upperquartile = int (np.percentile(gdpvals, 75))

for v in dictmap_natCntry.keys():
    if dictmap_natCntry[v] <= lowerquartile:
       dictmap_natCntry[v] = 1
    if dictmap_natCntry[v] <= median and dictmap_natCntry[v]>lowerquartile:
       dictmap_natCntry[v] = 2
    if dictmap_natCntry[v]>median and dictmap_natCntry[v]<=upperquartile:
        dictmap_natCntry[v] = 3
    if dictmap_natCntry[v]>upperquartile:
        dictmap_natCntry[v] = 4

TRAIN_DATA['native-country'] = TRAIN_DATA['native-country'].map(dictmap_natCntry)
ax = TRAIN_DATA['native-country'].plot.bar(x='native-country', y='val', rot=0)
TRAIN_DATA['native-country'].value_counts().plot(kind="bar", axes=ax)
numerical_attributes = TRAIN_DATA.select_dtypes(include=['object'])


dict = {}
cnt = 0
for index, row in TRAIN_DATA.iterrows():
    if row['capital-loss']>0:
        dict[row['capital-loss']]=row['capital-loss']
        cnt+=1
dict[0]=0
for v in dict.values():
    cLossvals.append(v)

lowerquartile = int (np.percentile(cLossvals,25))
median = int (np.percentile(cLossvals,50))
upperquartile = int (np.percentile(cLossvals,75))

for v in dict.keys():
    if dict[v] <= lowerquartile:
       dict[v] = 1
    if dict[v] <= median and dict[v]>lowerquartile:
       dict[v] = 2
    if dict[v]>median and dict[v]<=upperquartile:
        dict[v] = 3
    if dict[v]>upperquartile:
        dict[v] = 4

TRAIN_DATA['capital-loss'] = TRAIN_DATA['capital-loss'].map(dict)

dict2 = {}
cnt = 0
for index, row in TRAIN_DATA.iterrows():
    if row['capital-gain']>0:
        dict2[row['capital-gain']]=row['capital-gain']
        cnt+=1
dict2[0]=0
for v in dict2.values():
    cGainvals.append(v)

lowerquartile = int (np.percentile(cGainvals,25))
median = int (np.percentile(cGainvals,50))
upperquartile = int (np.percentile(cGainvals,75))

for v in dict2.keys():
    if dict2[v] <= lowerquartile:
       dict2[v] = 1
    if dict2[v] <= median and dict2[v]>lowerquartile:
       dict2[v] = 2
    if dict2[v]>median and dict2[v]<=upperquartile:
        dict2[v] = 3
    if dict2[v]>upperquartile:
        dict2[v] = 4

TRAIN_DATA['capital-gain'] = TRAIN_DATA['capital-gain'].map(dict2)
TRAIN_DATA = pd.get_dummies(TRAIN_DATA,columns=['occupation'],prefix=['occ'])
TRAIN_DATA.drop(['workClass','race','marital-status','relationship'],axis=1,inplace=True)

df1 = TRAIN_DATA.pop('income')
TRAIN_DATA['income'] = df1

# ******************************************************************************

# ********************************* TEST_DATA **********************************


TEST_DATA['workClass'].fillna(TEST_DATA['workClass'].value_counts().index[0], inplace=True)
TEST_DATA['occupation'].fillna(TEST_DATA['occupation'].value_counts().index[0], inplace=True)
TEST_DATA['native-country'].fillna(TEST_DATA['native-country'].value_counts().index[0], inplace=True)

df = TEST_DATA[TEST_DATA.duplicated(keep=False)]
TEST_DATA = TEST_DATA.drop_duplicates(keep='first')


dictmap_sex = {'Male': 0, 'Female': 1}
TEST_DATA['sex'] = TEST_DATA['sex'].map(dictmap_sex)

dictmap_edu = {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8,
               'Prof-school': 9, 'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college': 12, 'Bachelors': 13, 'Masters': 14, 'Doctorate': 15}
TEST_DATA['education'] = TEST_DATA['education'].map(dictmap_edu)

#  Not-working = 0, Gov = 7, Self-emp = 9, Private = 10
dictmap_wrkCl = {'Private': 'Private', 'Self-emp-not-inc': 'Self-emp', 'Self-emp-inc': 'Self-emp', 'Local-gov': 'Gov', 'State-gov': 'Gov', 'Federal-gov': 'Gov', 'Without-pay': 'Not-working', 'Never-worked': 'Not-working'}
TEST_DATA['workClass'] = TEST_DATA['workClass'].map(dictmap_wrkCl)

dictmap_Occ = {'Adm-clerical': 'Admin', 'Armed-Forces': 'Military', 'Craft-repair': 'Blue-Collar', 'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar', 'Handlers-cleaners': 'Blue-Collar', 'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service', 'Priv-house-serv': 'Service', 'Prof-specialty':'Professional', 'Protective-serv': 'Other-Occupations', 'Sales': 'Sales', 'Tech-support': 'Other-Occupations', 'Transport-moving': 'Blue-Collar'}
TEST_DATA['occupation'] = TEST_DATA['occupation'].map(dictmap_Occ)

dictmap_MarSt = {'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married', 'Married-civ-spouse': 'Married', 'Married-spouse-absent': 'Not-Married', 'Separated': 'Not-Married', 'Divorced': 'Not-Married', 'Widowed': 'Widowed'}
TEST_DATA['marital-status'] = TEST_DATA['marital-status'].map(dictmap_MarSt)

TEST_DATA['marital_relation'] = TEST_DATA['marital-status'] + ' ' + TEST_DATA['relationship']
TEST_DATA = pd.get_dummies(TEST_DATA,columns=['marital_relation'])


dictmap_natCntry = {'United-States':27776, 'Cuba':2621, 'Jamaica':2156, 'India':342, 'Mexico':5715, 'South':3447, 'Puerto-Rico':10876, 'Honduras':618, 'England':19709, 'Canada':19859, 'Germany':27087 , 'Iran':1202, 'Philippines':939,
                    'Italy':19273, 'Poland':2874, 'Columbia':2218, 'Cambodia':270, 'Thailand':2490, 'Ecuador':2028, 'Laos':325, 'Taiwan':3225, 'Haiti':282, 'Portugal':9978, 'Dominican-Republic':1891, 'El-Salvador':1384, 'France':23496,
                    'Guatemala':1276, 'China':473, 'Japan':39268, 'Yugoslavia':4250, 'Peru':1900, 'Outlying-US(Guam-USVI-etc)': 34954, 'Scotland':31367, 'Trinadad&Tobago':3956, 'Greece':11091, 'Nicaragua':854, 'Vietnam':220, 'Hong':22502,
                    'Ireland':12430, 'Hungary':4172, 'Holand-Netherlands':24331}

gdpvals=[]
for v in dictmap_natCntry.values():
    gdpvals.append(v)

lowerquartile = np.percentile(gdpvals, 25)
median = np.percentile(gdpvals, 50)
upperquartile = np.percentile(gdpvals, 75)


gdpvals=[]
cLossvals = []
cGainvals = []
for v in dictmap_natCntry.values():
    gdpvals.append(v)

lowerquartile = int (np.percentile(gdpvals, 25))
median = int (np.percentile(gdpvals, 50))
upperquartile = int (np.percentile(gdpvals, 75))


for v in dictmap_natCntry.keys():
    if dictmap_natCntry[v] <= lowerquartile:
       dictmap_natCntry[v] = 1
    if dictmap_natCntry[v] <= median and dictmap_natCntry[v]>lowerquartile:
       dictmap_natCntry[v] = 2
    if dictmap_natCntry[v]>median and dictmap_natCntry[v]<=upperquartile:
        dictmap_natCntry[v] = 3
    if dictmap_natCntry[v]>upperquartile:
        dictmap_natCntry[v] = 4

TEST_DATA['native-country'] = TEST_DATA['native-country'].map(dictmap_natCntry)

dict = {}
cnt = 0
for index, row in TEST_DATA.iterrows():
    if row['capital-loss']>0:
        dict[row['capital-loss']]=row['capital-loss']
        cnt+=1
dict[0]=0
for v in dict.values():
    cLossvals.append(v)


lowerquartile = int (np.percentile(cLossvals,25))
median = int (np.percentile(cLossvals,50))
upperquartile = int (np.percentile(cLossvals,75))

for v in dict.keys():
    if dict[v] <= lowerquartile:
       dict[v] = 1
    if dict[v] <= median and dict[v]>lowerquartile:
       dict[v] = 2
    if dict[v]>median and dict[v]<=upperquartile:
        dict[v] = 3
    if dict[v]>upperquartile:
        dict[v] = 4

TEST_DATA['capital-loss'] = TEST_DATA['capital-loss'].map(dict)

dict2 = {}
cnt = 0
for index, row in TEST_DATA.iterrows():
    if row['capital-gain']>0:
        dict2[row['capital-gain']]=row['capital-gain']
        cnt+=1
dict2[0]=0
for v in dict2.values():
    cGainvals.append(v)

lowerquartile = int (np.percentile(cGainvals,25))
median = int (np.percentile(cGainvals,50))
upperquartile = int (np.percentile(cGainvals,75))

for v in dict2.keys():
    if dict2[v] <= lowerquartile:
       dict2[v] = 1
    if dict2[v] <= median and dict2[v]>lowerquartile:
       dict2[v] = 2
    if dict2[v]>median and dict2[v]<=upperquartile:
        dict2[v] = 3
    if dict2[v]>upperquartile:
        dict2[v] = 4

TEST_DATA['capital-gain'] = TEST_DATA['capital-gain'].map(dict2)
TEST_DATA = pd.get_dummies(TEST_DATA,columns=['occupation'],prefix=['occ'])


TEST_DATA.drop(['workClass','race','marital-status','relationship'],axis=1,inplace=True)

# Moving 'income' column to last
df1 = TEST_DATA.pop('income')
TEST_DATA['income'] = df1

# ******************************************************************************

# ******************************* TRAINING MODEL *******************************

income_train = TRAIN_DATA[['income']].copy()
income_test = TEST_DATA[['income']].copy()

df_train = TRAIN_DATA.copy()
df_test = TEST_DATA.copy()
df_train = df_train.drop(['income'],axis=1,inplace=False)
df_test = df_test.drop(['income'],axis=1,inplace = False)

TRAIN_DATA_normalized = preprocessing.normalize(df_train)
TRAIN_DATA_normalized_df = pd.DataFrame(TRAIN_DATA_normalized,columns=TRAIN_DATA.columns[:34])

TEST_DATA_normalized = preprocessing.normalize(df_test)
TEST_DATA_normalized_df = pd.DataFrame(TEST_DATA_normalized,columns=TEST_DATA.columns[:34])

income_train = TRAIN_DATA[['income']].copy()
income_test = TEST_DATA[['income']].copy()

X_train = np.array(TRAIN_DATA_normalized_df)
income_train["income"] = income_train["income"].astype('category')
income_train["nincome"] = income_train["income"].cat.codes
income_train.drop(['income'],axis=1,inplace=True)
income_train.rename(columns={'nincome':'income'},inplace=True)
y_train = income_train

X_test = np.array(TEST_DATA_normalized_df)
income_test["income"] = income_test["income"].astype('category')
income_test["nincome"] = income_test["income"].cat.codes
income_test.drop(['income'],axis=1,inplace=True)
income_test.rename(columns={'nincome':'income'},inplace=True)
y_test = income_test


#clf = MultinomialNB()
# clf = LogisticRegression()
# clf = svm.SVC()
clf = neighbors.KNeighborsClassifier(n_neighbors = 180)
clf.fit(preprocessing.scale(X_train), y_train.values.ravel())
accuracy = clf.score(preprocessing.scale(X_test), y_test)

# Predicted class
print("KNN Accuracy = ",accuracy)

# ******************************************************************************

