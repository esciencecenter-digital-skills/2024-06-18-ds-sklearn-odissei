![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1 Introduction to Machine Learning (Day 2 Odissei summer school)

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/odissei-ml-day1

Collaborative Document day 1: https://tinyurl.com/odissei-ml-day1

Collaborative Document day 2: https://tinyurl.com/odissei-ml-day2


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## Links

### 🖥 Workshop website
https://esciencecenter-digital-skills.github.io/2024-06-18-ds-sklearn-odissei

### 🛠 Setup
https://esciencecenter-digital-skills.github.io/2024-06-18-ds-sklearn-odissei#setup

## 👩‍🏫👩‍💻🎓 Instructors

Sven van der Burg, Flavio Hafner, Malte Luken, Carsten Schnober


## 🗓️ Agenda
09:00	Welcome and icebreaker
09:15	Introduction to machine learning
10:00	Break
10:10	Tabular data exploration
10:40   First model with scikit-learn
11:00	Break
11:10	First model with scikit-learn
12:00	Lunch Break
13:00	Fitting a scikit-learn model on numerical data
13:45	Working with numerical data
14:20	Break
14:30   Model evaluation using cross-validation 
14:45	Handling categorical data
15:40	Intuition on tree-based models
15:50	Break
16:00	Guest lecture
17:00	END

## Welcome and icebreaker
- Icebreaker:
    - What did you have for breakfast this morning?
    - How is the energy?
    - Exposure to machine learning so far?
- Goal of this workshop: Quickly teach you the basics of machine learning, preparing you for applying machine learning on the LISS dataset.
- ⚠️Manage expectations⚠️: 
    - We start with the absolute basics, but will discuss more advanced topics as we progress.
    - We assume you have a statistical background, ask for clarification if we go to fast!
    - 

## 🏢 Location logistics
* Coffee?
* Wifi?
* Toilets?
* Emergency exit?

## 🔧 Exercises

### Exercise: Machine learning concepts [Sven]
Given a case study: pricing apartments based on a real estate website. We have thousands of house descriptions with their price. Typically, an example of a house description is the following:

“Great for entertaining: spacious, updated 2 bedroom, 1 bathroom apartment in Lakeview, 97630. The house will be available from May 1st. Close to nightlife with private backyard. Price ~$1,000,000.”

We are interested in predicting house prices from their description. One potential use case for this would be, as a buyer, to find houses that are cheap compared to their market value.

#### What kind of problem is it?

a) a supervised problem
b) an unsupervised problem
c) a classification problem
d) a regression problem

Select all answers that apply



#### What are the features?

a) the number of rooms might be a feature
b) the post code of the house might be a feature
c) the price of the house might be a feature

Select all answers that apply


#### What is the target variable?

a) the full text description is the target
b) the price of the house is the target
c) only house description with no price mentioned are the target

Select a single answer



#### What is a sample?

a) each house description is a sample
b) each house price is a sample
c) each kind of description (as the house size) is a sample

Select a single answer

#### (optional) Think of a machine learning task that would be relevant to solve in your research field. Try to answer the above questions for it.


### Exercise: Are you setup?
If you have not followed the setup instructions yet, please do so now by following [these instructions](https://github.com/INRIA/scikit-learn-mooc/blob/main/local-install-instructions.md)

#### Check your installation
* Open a command line terminal where you have access to your conda installation (Miniconda or Anaconda). On Windows this is `Anaconda prompt` or `Anaconda Powershell Prompt`. On Mac or Linux open a terminal.
* Navigate to the folder `scikit-learn-mooc` that you cloned/downloaded when following the setup instructions.
* Activate the environment: `conda activate scikit-learn-course`
* Check your environment: `python check_env.py`
* Put the orange sticky on your laptop lid to indicate you are ready!

#### (Alternative exercise if you are setup) Predicting Student Performance

You are interested in predicting student performance in school based on various socioeconomic and educational factors. You have access to a dataset that includes information about students' demographics, family background, and academic records.

**Dataset Features:**
1. **StudentID**: Unique identifier for each student.
2. **Gender**: Male or Female.
3. **Age**: Age of the student (years).
4. **ParentalEducation**: Highest level of education attained by the student's parents (categorical).
5. **FamilyIncome**: Annual family income (USD).
6. **StudyTime**: Weekly study time (hours).
7. **SchoolSupport**: Whether the student receives additional educational support (Yes or No).
8. **ExtraActivities**: Participation in extracurricular activities (Yes or No).
9. **InternetAccess**: Whether the student has internet access at home (Yes or No).
10. **Grades**: Average grades of the student in the last school year (numerical).

1. **Data Understanding:**
    - Describe the types of data you have. What are the different types of variables in this dataset (e.g., numerical, categorical)?
    - Which variable is the target variable?
    - Is this a supervised or unsupervised machine learning task?
    - Is this a regression or classifcation task?

2. **Feature Selection:**
    - Identify three features you think would be most important for predicting student performance. Explain your reasoning.

### 📝 Exercise : Adapting your first model
The goal of this exercise is to fit a similar model as we just did to get familiar with manipulating scikit-learn objects and in particular the `.fit/.predict/.score` API.

Before we used `model = KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the KNeighborsClassifier parameters is n_neighbors. It controls the number of neighbors we are going to use to make a prediction for a new data point.

#### 1. What is the default value of the n_neighbors parameter? 
Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This will open a pager pointing to the documentation.
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```

#### 2. Create a KNeighborsClassifier model with n_neighbors=50
a. Fit this model on the train data and target that we used before
b. Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?
c. Compute the accuracy on the training data.
d. Now load the test data from "../datasets/adult-census-numeric-test.csv" and compute the accuracy on the test data.

#### 3. (Optional) Find the optimal n_neighbors
What is the optimal number of neighbors to fit a K-neighbors classifier on this dataset?

### Exercise: Compare with simple baselines
#### 1. Compare with simple baseline
The goal of this exercise is to compare the performance of our classifier in the previous notebook (roughly 81% accuracy with LogisticRegression) to some simple baseline classifiers. The simplest baseline classifier is one that always predicts the same class, irrespective of the input data.

What would be the score of a model that always predicts ' >50K'?

What would be the score of a model that always predicts ' <=50K'?

Is 81% or 82% accuracy a good score for this problem?

Use a DummyClassifier such that the resulting classifier will always predict the class ' >50K'. What is the accuracy score on the test set? Repeat the experiment by always predicting the class ' <=50K'.

Hint: you can set the strategy parameter of the DummyClassifier to achieve the desired behavior.

You can import DummyClassifier like this:
```python
from sklearn.dummy import DummyClassifier
```

##### Solution

Compare to simple baseline:
```python
from sklearn.dummy import DummyClassifier

class_to_predict = " >50K"
```
```python
high_revenue_clf = DummyClassifier(
    strategy="constant", constant=class_to_predict
)
```
Fit baseline classifier to data:
```python
high_revenue_clf.fit(data_train, target_train)
```
Evaluate baseline classifier:
```python
high_revenue_clf.score(data_test, target_test)

0.23396937187781508
```

Use other class for baseline classifier:
```python
class_to_predict = " <=50K"

low_revenue_clf = DummyClassifier(
    strategy="constant", constant=class_to_predict
)

low_revenue_clf.fit(data_train, target_train)

low_revenue_clf.score(data_test, target_test)

0.7660306281221849
```
The logistic regression qualifier above performs only slightly better than the `low_revenue classifier` (~0.82 vs. ~0.77).

The interpretation of these scores depends on the data, for instance increases are more meaningful on small data size.

#### 2. (optional) Try out other baselines
What other baselines can you think of? How well do they perform?

* Ignacio: 1. 23%, 77%; 2. Random assignment (Score = 0.5)
* Yael: 
 ```python
dummy_clf = DummyClassifier(strategy="constant", constant=" >50K")
dummy_clf.fit(data_train, target_train)
output2 = dummy_clf.score(data_test, target_test)
output2
```
0.23396937187781508

### Exercise: Recap fitting a scikit-learn model on numerical data

#### 1. Why do we need two sets: a train set and a test set?

a) to train the model faster
b) to validate the model on unseen data
c) to improve the accuracy of the model

Select all answers that apply

#### 2. The generalization performance of a scikit-learn model can be evaluated by:

a) calling fit to train the model on the training set, predict on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function
b) calling fit to train the model on the training set and score to compute the score on the test set
c) calling cross_validate by passing the model, the data and the target
d) calling fit_transform on the data and then score to compute the score on the test set

Select all answers that apply

#### 3. When calling `cross_validate(estimator, X, y, cv=5)`, the following happens:

a) X and y are internally split five times with non-overlapping test sets
b) estimator.fit is called 5 times on the full X and y
c) estimator.fit is called 5 times, each time on a different training set
d) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the train sets
e) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the test sets

Select all answers that apply

#### 4. (optional) Scaling
We define a 2-dimensional dataset represented graphically as follows:
![](https://i.imgur.com/muvSbI6.png)

Question

If we process the dataset using a StandardScaler with the default parameters, which of the following results do you expect:

![](https://i.imgur.com/t5mTlVG.png)


a) Preprocessing A
b) Preprocessing B
c) Preprocessing C
d) Preprocessing D

Select a single answer

#### 5. (optional) Cross-validation allows us to:

a) train the model faster
b) measure the generalization performance of the model
c) reach better generalization performance
d) estimate the variability of the generalization score

Select all answers that apply


### Solutions

1: b
2: a,b,c
3: a,c,e
4: a,b
5: b,c,(d)

## 🧠 Collaborative Notes

### Introduction to Machine Learning

Goal: Building predictive models that generalize to new instances

Examples:
- Predict iris (flower) type based on petal and sepal length
- Predict income class based on household characteristics

#### Generalizing vs. memorizing
Generalizing: Predict new instances based on systematic rules inferred from sources of variability in the data set

Memorizing: Store all known instances and predict based on closest match between known instances and new instance

Training data: Data set used to learn systematic rules and encapsulate them in the predictive model

Test data: Data set used to assess how well the model can generalize to unseen instances

Data matrix (denoted *X*):
- Rows are observations or samples
- Columns are features
- Input for the model

Target (denoted *y*):
- A variable for each observation to predict with the model

The goal is finding a structure in data *X* to make good predictions for target *y*

#### Regression vs Classification

Regression: When the target is a continuous variable (e.g., the price of a house)

Classification: When the target is a discrete variable (e.g., income class)

### Data Exploration

#### Opening Jupyter Lab

Navigate to `scicit-learn-mooc` with `cd` and type `jupyter lab`

#### Importing the Data

Import the data frame:
```python
# Import package for data manipulation
import pandas as pd

# Import data set from CSV file
adult_census = pd.read_csv("datasets/adult_census.csv")

# Get quick overview of data by printing first 5 rows
adult_census.head()

# Get number of samples and number of features
n_rows, n_cols = adult_census.shape
```

Change a cell from "Code" to "Markdown" by hitting `Esc` and `M`

#### Inspecting the Data

Get and overview of the distribution of the features

This is important for understanding the representativeness of the data and applying potential transformations

```python
# Plot a histogram of individual features
_ = adult_census.hist(figsize=(20, 14))
```

#### Separating the Data Matrix and the Target

```python
# Set name of target variable
target_column = "class"

# Count unique values of target
adult_census[target_column].value_counts()
```

Imbalance in the target values can must be taken into account when assessing the performance of machine learning models

E.g., when once class has a much higher count than the other, the model might always predict this class achieving a high accuracy despite not having learned to distinguish the classes

#### Inspecting Relationships Between Features

```python
# Import package for plotting
import seaborn as sns

# Only use first 5000 rows
n_samples = 5000
# Only use numeric variables
columns = ["age", "education-num", "hours-per-week"]

# Create pairplot of numeric variables using first 5000 rows
sns.pairplot(
    data=adult_census[:n_samples],
    vars=columns,
    hue=target_column,
    diag_kind="hist",
    diag_kws={"bins": 30}
)
```

### Training the First Machine Learning Model

Import the data frame:
```python
# Import package for data manipulation
import pandas as pd

# Import data set with only numeric features from CSV file
adult_census = pd.read_csv("datasets/adult-census-numeric.csv")

# Get a quick overview of data
adult_census.head()

# Set name of target variable
target_name = "class"

# Select target from data set
target = adult_census[target_name]

# Remove target from data set to get data matrix
data = adult_census.drop(columns=[target_name])

# Look at feature names and number of samples, features
data.columns
data.shape
```

We use the k-nearest neighbor classifier to predict the class based on numeric features

The `fit` method takes data matrix and target as input to train the model

```python
# Import model from scikit-learn package
from sklearn.neihhbors import KNeighborsClassifier

# Create a new model instance
model = KNeighborsClassifier()
# Fit the model to data matrix and target
model.fit(data, target)
```

The `predict` method predicts the target for new data

But first, me make predictions for the training data

```python
# Predict target for training data matrix
target_predicted = model.predict(data)

# Check if predictions are correct for first 5 samples
target[:5] == target_predicted[:5]

# Check predictions for entire data set
target == target_predicted

# Calcuate accuracy (average correctness)
(target == target_predicted).mean()

```

To assess the predictive performance, we need to test the model on the test data set

```python
# Load test data set
adult_census_test = pd.read_csv("datasets/adult-census-numeric-test.csv")

# Separate target and data matrix
target_test = adult_census_test[target_name]

data_test = adult_census_test.drop(columns=[target_name])

# Predict for test data
model.predict(data_test)

```

The `score` method can be used to make predictions and compute accuracy in one go

```python
# Calculate accuracy on test data
model.score(data_test, target_test)
```

The methods `fit`, `predict`, `score` are the core of the scikit-learn machine learning workflow

Inspect the parameters of a machine learning model: `?KNeighborsClassifier`

### Working with numerical data

```python
import pandas as pd

adult_census = pd.read_csv("datasets/adult-census.csv")
```
We ignore the `education-num` column:
```python
adult_census = adult_census.drop(columns=["education-num"])
adult_census.head()
Skip to main panel
>
import pandas as pd

adult_census = pd.read_csv("datasets/adult-census.csv")

adult_census = adult_census.drop(columns=["education-num"])
adult_census.head()

	age 	workclass 	education 	marital-status 	occupation 	relationship 	race 	sex 	capital-gain 	capital-loss 	hours-per-week 	native-country 	class
0 	25 	Private 	11th 	Never-married 	Machine-op-inspct 	Own-child 	Black 	Male 	0 	0 	40 	United-States 	<=50K
1 	38 	Private 	HS-grad 	Married-civ-spouse 	Farming-fishing 	Husband 	White 	Male 	0 	0 	50 	United-States 	<=50K
2 	28 	Local-gov 	Assoc-acdm 	Married-civ-spouse 	Protective-serv 	Husband 	White 	Male 	0 	0 	40 	United-States 	>50K
3 	44 	Private 	Some-college 	Married-civ-spouse 	Machine-op-inspct 	Husband 	Black 	Male 	7688 	0 	40 	United-States 	>50K
4 	18 	? 	Some-college 	Never-married 	? 	Own-child 	White 	Female 	0 	0 	30 	United-States 	<=50K
```
Separate data from target class, using the same logic as previously but in a single line:
```python
data, target = adult_census.drop(columns=["class"]), adult_census["class"]
```
Look at the data:
```python
data.head()

 	age 	workclass 	education 	marital-status 	occupation 	relationship 	race 	sex 	capital-gain 	capital-loss 	hours-per-week 	native-country
0 	25 	Private 	11th 	Never-married 	Machine-op-inspct 	Own-child 	Black 	Male 	0 	0 	40 	United-States
1 	38 	Private 	HS-grad 	Married-civ-spouse 	Farming-fishing 	Husband 	White 	Male 	0 	0 	50 	United-States
2 	28 	Local-gov 	Assoc-acdm 	Married-civ-spouse 	Protective-serv 	Husband 	White 	Male 	0 	0 	40 	United-States
3 	44 	Private 	Some-college 	Married-civ-spouse 	Machine-op-inspct 	Husband 	Black 	Male 	7688 	0 	40 	United-States
4 	18 	? 	Some-college 	Never-married 	? 	Own-child 	White 	Female 	0 	0 	30 	United-States
```

```python
data.dtypes

age                int64
workclass         object
education         object
marital-status    object
occupation        object
relationship      object
race              object
sex               object
capital-gain       int64
capital-loss       int64
hours-per-week     int64
native-country    object
dtype: object
```
Categorical variables are usually represented as `object` type, but not necessarily.

All data types supported by NumPy are listed in the [documentation](https://numpy.org/doc/stable/user/basics.types.html).

`dtype` is called without parentheses because it is an attribute. Functions, however, need to be called with parantheses, for instance `head()`.

Define numerical columns:
```python
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
```
This works independently of the order of the column list, but the output order adjusts accordingly.

Take subset of numerical from data:
```python
data[numerical_columns].head()

 	age 	capital-gain 	capital-loss 	hours-per-week
0 	25 	0 	0 	40
1 	38 	0 	0 	50
2 	28 	0 	0 	40
3 	44 	7688 	0 	40
4 	18 	0 	0 	30
```
```python
data_numeric = data[numerical_columns]
```

Describe numeric data variables, for instance the `age` column:
```python
data_numeric["age"].describe()

count    48842.000000
mean        38.643585
std         13.710510
min         17.000000
25%         28.000000
50%         37.000000
75%         48.000000
max         90.000000
Name: age, dtype: float64
```
#### Split data into train and test set

Instead of loading a separate dataset, we split the entire dataset:
```python
from sklearn.model_selection import train_test_split
```
Split the data randomly
```python
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25
)
```
In order to make the random split reproducible, a random state (seed) is predefined with `random_state`.

The size of the test set, defined as 25% of the total data with `test_size=0.25` here, should be large enough to be representative while leaving enough data for training the model.

Get the size of the training and the data set:
```python
data_train.shape
(36631, 4)
```
```python
data_test.shape
(12211, 4)
```

#### Fitting a logistic regression model

Logistic regression models are (confusingly) used for classification too.

```python
from sklearn.linear_model import LogisticRegression
```

Define new instance of a logistic regression model:
```python
model = LogisticRegression()
```

Fit the model to our data:
```python
model.fit(data_train, target_train)
```
```python
accuracy = model.score(data_test, target_test)
accuracy

0.8070592089099992
```

### Scaling numerical features

Look at the numerical data:
```python
data_train.describe()

 	age 	capital-gain 	capital-loss 	hours-per-week
count 	36631.000000 	36631.000000 	36631.000000 	36631.000000
mean 	38.642352 	1087.077721 	89.665311 	40.431247
std 	13.725748 	7522.692939 	407.110175 	12.423952
min 	17.000000 	0.000000 	0.000000 	1.000000
25% 	28.000000 	0.000000 	0.000000 	40.000000
50% 	37.000000 	0.000000 	0.000000 	40.000000
75% 	48.000000 	0.000000 	0.000000 	45.000000
max 	90.000000 	99999.000000 	4356.000000 	
```

The variables are not comparable with each other because they have different scales of magnitudes.
Depending on the model, this affects accuracy and/or the training duration.
For large datasets, the latter is relevant.
For Decision Tree models, scale does not matter.

Scikit-Learn provides a tool for scaling:
```python
from sklearn.preprocessing import StandardScaler
```
Create a new scaler:
```python
scaler = StandardScaler()
```
A scaler has similar methods as a machine learning model, including `.fit()`:
```python
scaler.fit(data_train)
```
As a result, each feature will have a standard deviation between 0 and 1.
This assumes roughly normally distributed variables. However, this is not always the case, e.g. for `capital-gain`.

Get the mean for each feature in the data:
```python
scaler.mean_

array([  38.64235211, 1087.07772106,   89.6653108 ,   40.43124676])
```

The scaler needs to be fit to the training set, and applied to the test set as well.
It must not be adapted to the test set to prevent leaking between training and test data.
The model must not learn anything about the test data so that the evaluation of the model is realistic.

Scale the training data:
```python
data_train_scaled = scaler.transform(data_train)
data_train_scaled

array([[ 0.17177061, -0.14450843,  5.71188483, -2.28845333],
       [ 0.02605707, -0.14450843, -0.22025127, -0.27618374],
       [-0.33822677, -0.14450843, -0.22025127,  0.77019645],
       ...,
       [-0.77536738, -0.14450843, -0.22025127, -0.03471139],
       [ 0.53605445, -0.14450843, -0.22025127, -0.03471139],
       [ 1.48319243, -0.14450843, -0.22025127, -2.69090725]])
```
Shortcut for fitting and transformation at once:
```python
data_train_scaled = scaler.fit_transform(data_train)
```

Instead of a NumPy array (default), make the result of `transform()` a Pandas dataframe:

```python
scaler = StandardScaler().set_output(transform="pandas")
```
Refit and transform the training data:
```python
data_train_scaled = scaler.fit_transform(data_train)
```

The result is a Pandas dataframe, so it has the `describe()` method:
```python
data_train_scaled.describe()

 	age 	capital-gain 	capital-loss 	hours-per-week
count 	3.663100e+04 	3.663100e+04 	3.663100e+04 	3.663100e+04
mean 	-2.273364e-16 	3.530310e-17 	3.840667e-17 	1.844684e-16
std 	1.000014e+00 	1.000014e+00 	1.000014e+00 	1.000014e+00
min 	-1.576792e+00 	-1.445084e-01 	-2.202513e-01 	-3.173852e+00
25% 	-7.753674e-01 	-1.445084e-01 	-2.202513e-01 	-3.471139e-02
50% 	-1.196565e-01 	-1.445084e-01 	-2.202513e-01 	-3.471139e-02
75% 	6.817680e-01 	-1.445084e-01 	-2.202513e-01 	3.677425e-01
max 	3.741752e+00 	1.314865e+01 	1.047970e+01 	4.714245e+00
```

### The machine learning pipeline

Scikit-Learn provides pipelines to combine multiple steps:
```python
from sklearn.pipeline import make_pipeline
```
For instance, combine a scaler and a logistic regression model:
```python
model = make_pipeline(StandardScaler(), LogisticRegression())
```
The pipeline can be visualized:
```python
model
```
The pipeline works like a machine learning model which always applies the scaler in addition.

Train and score the model again:
```python
model.fit(data_train, target_train)
model.score(data_test, target_test)

0.8071411022848252
```

Other pre-processing methods are provided by scikit-learn, using the same syntax.

### Model evaluation using cross-validation

The train-test split has an effect on the accuracy.

Cross-validation folds the entire data set multiple times, each time using a different train-test split.
We get a different accuracy for each fold. The mean and the variance give an idea of the model performance.

With `shuffle=True`, the train-test split is random instead of sequentially iterating over the data.

```python
from sklearn.model_selection import cross_validate

cv_result = cross_validate(model, data_numeric, target, cv=5)
```
```python
cv_result

{'fit_time': array([0.04306483, 0.03754997, 0.03796721, 0.03651094, 0.03571296]),
 'score_time': array([0.00925016, 0.00913596, 0.00918174, 0.00873089, 0.00879407]),
 'test_score': array([0.79557785, 0.80049135, 0.79965192, 0.79873055, 0.80456593])}
```

### Handling categorical data

```python
from sklearn.compose import make_column_selector as selector
```
Automatically select all categorical variable columns, assuming they are all of type `object`:
```python
categorical_column_selector = selector(dtype_include=object)
```
```python
categorical_columns = categorical_column_selector(data)
categorical_columns

['workclass',
 'education',
 'marital-status',
 'occupation',
 'relationship',
 'race',
 'sex',
 'native-country']
```
Create new dataset by selecting subset of categorical columns:
```python
data_categorical = data[categorical_columns]
data_categorical.head()

workclass 	education 	marital-status 	occupation 	relationship 	race 	sex 	native-country
0 	Private 	11th 	Never-married 	Machine-op-inspct 	Own-child 	Black 	Male 	United-States
1 	Private 	HS-grad 	Married-civ-spouse 	Farming-fishing 	Husband 	White 	Male 	United-States
2 	Local-gov 	Assoc-acdm 	Married-civ-spouse 	Protective-serv 	Husband 	White 	Male 	United-States
3 	Private 	Some-college 	Married-civ-spouse 	Machine-op-inspct 	Husband 	Black 	Male 	United-States
4 	? 	Some-college 	Never-married 	? 	Own-child 	White 	Female 	United-States
```

#### Strategies to encode categorical variables

- Ordinal encoding: convert into numbers, e.g. 1-4
- One-hot encoding/dummy variables: 0 or 1

##### Encoding ordinal categories

```python
from sklearn.preprocessing import OrdinalEncoder
```

Extract `education` column:
```python
education_column = data_categorical[["education"]]
```
Initialise encoder:
```python
encoder = OrdinalEncoder().set_output(transform="pandas")
```
Fit and transform education column:
```python
education_encoded = encoder.fit_transform(education_column)
education_encoded

education
0 	1.0
1 	11.0
2 	7.0
3 	15.0
4 	15.0
```

Compare to input before transforming:
```python
education_column.head()

 	education
0 	11th
1 	HS-grad
2 	Assoc-acdm
3 	Some-college
4 	Some-college
```
Get all categories:
```python
encoder.categories_

[array([' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th',
        ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate',
        ' HS-grad', ' Masters', ' Preschool', ' Prof-school',
        ' Some-college'], dtype=object)]
```
Note: categories are often inconsistent, e.g. `10th` vs. `1st-4th`.

The `list` parameter can be used to manually define the categories when initialising the `OrdinalEncoder`.

In the same way as for the `education` column, pass the entire categorical dataset:
```python
data_encoded = encoder.fit_transform(data_categorical)
data_encoded.head()

workclass 	education 	marital-status 	occupation 	relationship 	race 	sex 	native-country
0 	4.0 	1.0 	4.0 	7.0 	3.0 	2.0 	1.0 	39.0
1 	4.0 	11.0 	2.0 	5.0 	0.0 	4.0 	1.0 	39.0
2 	2.0 	7.0 	2.0 	11.0 	0.0 	4.0 	1.0 	39.0
3 	4.0 	15.0 	2.0 	7.0 	0.0 	2.0 	1.0 	39.0
4 	0.0 	15.0 	4.0 	0.0 	3.0 	4.0 	0.0 	39.0
```
#### One-hot encoding

```python
from sklearn.preprocessing import OneHotEncoder
```
Initialise as previously, but preventing sparse output:
```python
encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
```

Fit and transform:
```python
education_encoded = encoder.fit_transform(education_column)
education_encoded.head()

education_ 10th 	education_ 11th 	education_ 12th 	education_ 1st-4th 	education_ 5th-6th 	education_ 7th-8th 	education_ 9th 	education_ Assoc-acdm 	education_ Assoc-voc 	education_ Bachelors 	education_ Doctorate 	education_ HS-grad 	education_ Masters 	education_ Preschool 	education_ Prof-school 	education_ Some-college
0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0
1 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0
2 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0
3 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0
4 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0
```
This has created a distinct column for each category, containing `1` or `0`.

Again, pass the entire categorical data:
```python
data_encoded = encoder.fit_transform(data_categorical)
```

This increases the data width to 102:
```python
data_encoded.shape
(48842, 102)
```
Compare to 8 columns in the input data:
```python
data_categorical.shape
(48842, 8)
```

Some models have a disadvantage from more higher number of columns, e.g. Decision Tree.
The advantage is that some models achieve higher accuracy than with ordinal encoding.

Rule of thumb:
- ordinal encoding for classification
- One Hot encoding for regression

## 💬Feedback
### Feedback morning
Can you give us feedback on the workshop so far? Please mention
(at least) one thing that went well and one thing that can be improved.
Think of the pace, the instructors, balance between exercises and live coding, the room, anything!

#### 👍 What went well?
*  Very slear explanations and good support
*  Nice team with plenty support and enough people
* Enough instructors to help out
* Clear and good to have different people to support and give slides; live notes work really well
* It is nice that the code is immediately added to the Collaborative Document
* Good support and good pace
*
* Clear answers to questions
* Very friendly and calm support. I admire your patience at times. 
* Nice team and very friendly people. The collaborative document is awesome, and it is quite fast-paced
*
*
*
*

#### 👎 What can be improved?
* Maybe next time installation can be done in the break
* Setup during the introduction?
*
* Setup was given as homework in the Python course. Perhaps clear(er) communication that you'd need to be set-up before the start of the course, could have incentivised people to email/call for help earlier.
*
*
* Sometimes it's hard to follow why and when you define something the way you do
* It sometimes feels a bit like a black-box now. 
* Pace sometimes seems slow and other times a bit too fast.
* Maybe some more explanation of what each code exactly does (typed within the code to get a nice overview of code + explanation of code in the notebooks)
* More annotations in the code would be useful

#### Sven's summary
* Setup issues are unfortunately unavoidable, but the instructions definitely need improvement. We wrote down how we think it can be simplified.
* We do not want to do the setup in the break, it would mean that the people that are already confused with the setup do not have a break.
* How can we more clearly communicate that you have to be setup before the workshop?
* We will try to explain better what certain code does, also ask for clarification if it is unclear!

### Feedback afternoon
Can you give us feedback on the workshop so far? Please mention
(at least) one thing that went well and one thing that can be improved.
Think of the pace, the instructors, balance between exercises and live coding, the room, anything!

#### 👍 What went well?
* I appreciated learning about the pipelines. I found the exercises useful. 
* Very clear explanation and step-by-step.
* Clear explanations, good efforts to make it understandable
* Clear explanations, nice pace of coding, admire your calm and level of engagement given the energy-level haha
* Everything was clear, easy to follow even with no python experience
* Very useful to learn the pipelines. 
* 
* Understand the basics for performing classifications.
* variety of topics covered and clear notetaking


#### 👎 What can be improved?
* Not sure how to improve, but a lot for one day. 
* Maybe a bit more explanation about attributes, objects, instances and methods.
* Would be nice to have more time to work on exercises, in the end if we don't practice info gets lost but maybe we have time tomorrow :)
* Lot to take in in one day, noticed that my attention was lacking towards the end
* same, attention is lacking and made typos a lot
* Make it more interactive or with longer exercises so we can be more engaged and level our energy levels up. 
* It would be nice to go a bit deeper for the people who already know some machine learning
* 
* 
* Speed a bit slow.

#### Sven's summary
- We realize we threw a lot of information at you in one day.
- Today we will put that information to practice and there is plenty of time to let sink in all that information.
- More time for exercises: Today we we have one big exercise. People that have some machine learning experience will also be challenged!

## 📚 Resources
