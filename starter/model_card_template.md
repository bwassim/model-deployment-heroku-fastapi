# Model Card 

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
### Solve a classification problem to predict if income is greater or less than $50k/yr in census data

## Model Details
Developed by O.B for Udacity Machine Learning DevOps Nanodegree
Type: Logistic Regression model 
* Hyper parameter:  hyperparameters: {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [400, 1000]}
* Best parameter: C:0.1, max_iter:400

## Intended Use
* Baseline model for the end-to-end pipeline using DVC, GithubAction, Heroku and FastAPI

## Training Data
We used the entire census income data set https://archive.ics.uci.edu/ml/datasets/census+income

Data columns (total 15 columns):
``` 
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0    age             32561 non-null  int64 
 1    workclass       32561 non-null  object
 2    fnlgt           32561 non-null  int64 
 3    education       32561 non-null  object
 4    education-num   32561 non-null  int64 
 5    marital-status  32561 non-null  object
 6    occupation      32561 non-null  object
 7    relationship    32561 non-null  object
 8    race            32561 non-null  object
 9    sex             32561 non-null  object
 10   capital-gain    32561 non-null  int64 
 11   capital-loss    32561 non-null  int64 
 12   hours-per-week  32561 non-null  int64 
 13   native-country  32561 non-null  object
 14   salary          32561 non-null  object
dtypes: int64(6), object(9)
memory usage: 3.7+ MB
```
- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
- salary: >50K, <=50K.

**Relevant Papers:** 

Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996

## Evaluation Data
After cleaning the data 20% only was used for evaluation 
.The remaining of the data was used for training the model with cross validation
## Metrics
To evaluate model predictions the following metrics were used with the following results
* precision: 0.71
* recall: 0.26
* fbeta: 0.38

### Remark:
Note that we can certainly improve the above results by using Random Forest classifiers, however
the focus here is more on the deployment of the complete pipeline until serving with Heroku
## Ethical Considerations
The data has features including sex, race and relationships status

## Caveats and Recommendations
It's important to note that the dataset is imbalanced since the distribution of salaries is given as follows 
``` 
- <=50K:        ~76%
- >50K          ~24%
```