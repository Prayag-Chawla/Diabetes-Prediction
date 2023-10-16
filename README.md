
## Diabetes Prediction
In this project, I have done EDA and plotted various graphs depicting whether a person has diabetes or not. I have used methods like clustering, elbow method, shap explainer model. various probablity techniues like P test and T tests(hypothesis tets) to get to the accurate results.

## Hypothesis test
The process of hypothesis testing is to draw inferences or some conclusion about the overall population or data by conducting some statistical tests on a sample. The same inferences are drawn for different machine learning models through T-test which I will discuss in this tutorial.

For drawing some inferences, we have to make some assumptions that lead to two terms that are used in the hypothesis testing.

Null hypothesis: It is regarding the assumption that there is no anomaly pattern or believing according to the assumption made.

Alternate hypothesis: Contrary to the null hypothesis, it shows that observation is the result of real effect.

A lot of different approaches are present to hypothesis testing of two models like creating two models on the features available with us. One model comprises all the features and the other with one less. So we can test the significance of individual features. However feature inter-dependency affect such simple methods.

The steps involved in the hypothesis testing are as follow:

Assume a null hypothesis, usually in machine learning algorithms we consider that there is no anomaly between the target and independent variable.

Collect a sample

Calculate test statistics

Decide either to accept or reject the null hypothesis


calculating test statistics:

T = (Mean - Claim)/ (Standard deviation / Sample Size^(1/2))

Which is -4.3818 after putting all the numbers.

Now we calculate t value for 0.05 significance and degree of freedom.

Note: Degree of Freedom = Sample Size - 1

From T table the value will be -1.699.

After comparison, we can see that the generated statistics are less than the statistics of the desired level of significance. So we can reject the claim made.

You can calculate the t value using stats.t.ppf() function of stats class of scipy library.
In regression problems, we generally follow the rule of P value, the feature which violates the significance level are removed, thus iteratively improving the model.

## CATboost classifier
CatBoost is a supervised machine learning method that is used by the Train Using AutoML tool and uses decision trees for classification and regression. As its name suggests, CatBoost has two main features, it works with categorical data (the Cat) and it uses gradient boosting (the Boost).

## Shap Explainer Model
SHAP values are a common way of getting a consistent and objective explanation of how each feature impacts the model's prediction.

SHAP values are based on game theory and assign an importance value to each feature in a model. Features with positive SHAP values positively impact the prediction, while those with negative values have a negative impact. The magnitude is a measure of how strong the effect is.

SHAP values are model-agnostic, meaning they can be used to interpret any machine learning model, including:

Linear regression
Decision trees
Random forests
Gradient boosting models
Neural networks


## Model used
The primary model has beeen CATboost classifier.
Thereafter, we have used SHAP explainer model to se which parameters have the most value.

## Libraries and Usage

```
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
# Import Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates, andrews_curves, radviz
# Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

```






## Accuracy
There was a very high Accuracy from the model as we were able to get the decision variables from the common area under the plotted graph.





## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
In the real world, this project is used in biomedical industries extensively. Prediction of a disease has a direct relation with synthetic data generation which is used in generative adversarial networks(GANs), and hence is very important in today's world.
## Appendix

A very crucial project in the realm of data science and bio medical domain using visualization techniques as well as machine learning modelling.

## Tech Stack

**Client:** Python, CATboost classifier, EDA analysis, machine learning, sequential model of ML, SHAP explainer model, data visualization libraries of python.



## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

