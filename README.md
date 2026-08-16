# Titanic Survival Prediction — End-to-End Machine Learning Project

This project focuses on predicting whether a passenger survived the Titanic disaster using demographic, ticket, and family-related information.

The project covers the complete machine learning workflow, including data exploration, missing value handling, feature engineering, data preprocessing, model comparison, hyperparameter tuning, and model evaluation.

## Table of Contents

- Project Overview
- Dataset
- Exploratory Data Analysis
- Missing Values
- Outlier Analysis
- Feature Engineering and Preprocessing
- Machine Learning Models
- Model Results
- Feature Importance
- Key Insights
- Technologies Used
- How to Run the Project
- License

## Project Overview

The objective of this project is to develop a binary classification model that predicts whether a Titanic passenger survived or did not survive.

The target variable is:

- `0`: Did not survive
- `1`: Survived

The prediction is based on several passenger characteristics, including passenger class, sex, age, fare, embarkation port, and family-related information.

The project follows a complete and reproducible machine learning process:

- Exploring the dataset
- Analyzing relationships between variables
- Handling missing values
- Creating new features
- Applying numerical and categorical preprocessing
- Comparing different machine learning algorithms
- Tuning the selected model
- Evaluating the final model on test data
- Interpreting feature importance

## Dataset

The dataset contains 891 observations and 12 variables.

The main variables in the dataset are:

- `PassengerId`: Unique identification number of each passenger
- `Survived`: Target variable indicating whether the passenger survived
- `Pclass`: Passenger ticket class
- `Name`: Passenger name
- `Sex`: Passenger gender
- `Age`: Passenger age
- `SibSp`: Number of siblings or spouses aboard the Titanic
- `Parch`: Number of parents or children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port where the passenger boarded the Titanic

The target variable is not evenly distributed. Approximately 38.4% of the passengers in the dataset survived, while the remaining passengers did not survive.

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure of the data and identify important relationships between the features and the target variable.

The analysis showed that age values were mostly concentrated between 20 and 40 years. The age distribution was slightly right-skewed because of older passengers.

Sex was one of the most important variables associated with survival. Female passengers had a higher survival rate compared to male passengers.

Passenger class was also strongly related to survival. First-class passengers had a higher chance of survival than second-class and third-class passengers. Third-class passengers had the lowest survival rate.

The fare variable also provided useful information. Passengers who paid higher fares generally belonged to higher passenger classes and had better survival rates.

The embarkation port showed some differences in survival rates. However, this relationship may be influenced by other variables such as passenger class and fare.

## Missing Values

The dataset contains missing values in several columns.

The `Cabin` column contains a very high percentage of missing values. Because of this, the column was removed from the analysis.

The missing values in the `Age` column were completed using the median age. Median imputation was preferred because it is less affected by extreme values.

The missing values in the `Embarked` column were completed using the most frequent category.

The imputation operations were implemented inside machine learning pipelines. This approach helps prevent data leakage between the training and test datasets.

## Outlier Analysis

Outlier analysis was performed for numerical variables such as `Age` and `Fare`.

Some passengers had unusually high fares, and there were also passengers who were older than the majority of the dataset. These values were not removed because they represent valid observations rather than data entry errors.

For example, a very high fare may represent a legitimate first-class ticket. Similarly, older passengers are valid members of the passenger population.

## Feature Engineering and Preprocessing

Feature engineering was used to create additional variables that could improve the predictive performance of the models.

The following features were created:

- `Title`: Extracted from the passenger's name. Examples include `Mr`, `Mrs`, `Miss`, and `Rare`.
- `FamilySize`: Represents the total number of family members traveling with the passenger.
- `IsAlone`: Indicates whether the passenger was traveling alone.

The `Title` feature was especially useful because it contains information related to gender, age, and social status.

Numerical features were processed using median imputation and standardization.

Categorical features were processed using most-frequent-value imputation and one-hot encoding.

The preprocessing steps were combined using `ColumnTransformer` and `Pipeline`. This made the workflow more organized, reproducible, and resistant to data leakage.

## Machine Learning Models

Several classification algorithms were evaluated during the project.

The following models were compared:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The models were evaluated using cross-validation in order to obtain more reliable performance estimates.

After comparing the models, the best-performing candidate model was selected for further optimization. Hyperparameters were tuned using `GridSearchCV`.

## Model Results

The final model achieved an accuracy of 81.56% on the test dataset.

For the passengers who survived, the model achieved:

- Precision: 81.03%
- Recall: 68.12%
- F1-score: 74.02%

For the passengers who did not survive, the model achieved:

- F1-score: 85.71%

The model was more successful at identifying passengers who did not survive than passengers who survived.

The recall value for the survived class indicates that some passengers who actually survived were incorrectly predicted as non-survivors. This shows that there is still room for improvement, especially in identifying all surviving passengers.

## Feature Importance

Feature importance analysis was used to understand which variables contributed most to the model's predictions.

The most important features included:

- `Title_Mr`
- `Fare`
- `Pclass_3`
- `Age`
- `FamilySize`
- `Title_Rare`
- `Sex_male`
- `Embarked_S`

The `Title_Mr` feature was the most important feature according to the model. This feature may contain information about the passenger's gender, age group, and social status.

The `Fare` feature was also highly important. Fare is related to passenger class and may reflect the passenger's socioeconomic status.

The `Pclass_3` feature was another important variable. Traveling in third class was generally associated with a lower probability of survival.

Age and family size also contributed to the model's predictions. These variables helped the model identify differences between children, adults, older passengers, and passengers traveling with family members.

## Key Insights

The analysis produced several important conclusions.

Sex was one of the strongest predictors of survival. Female passengers had a substantially higher probability of surviving than male passengers.

Passenger class and fare were also important predictors. Passengers from higher classes and those who paid higher fares generally had better survival outcomes.

Feature engineering improved the representation of the data. The `Title`, `FamilySize`, and `IsAlone` features provided additional information that was not directly available from the original variables.

The `Title` feature was particularly useful because it combined information about a passenger's name, gender, age, and social status.

The use of `Pipeline` and `ColumnTransformer` created a consistent preprocessing process and helped prevent data leakage.

Although the model achieved an accuracy of 81.56%, its recall for the survived class was 68.12%. Future improvements could include class weighting, probability threshold optimization, additional ensemble methods, or further feature engineering.

## Technologies Used

This project was developed using the following technologies and libraries:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

The Scikit-learn tools used in the project include:

- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `GridSearchCV`
- Classification algorithms

## How to Run the Project

First, clone the repository:

```bash
git clone https://github.com/kozyurt/Titanic.git
