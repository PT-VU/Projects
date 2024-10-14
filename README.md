<div align="center">
<img width="674" alt="image" src="https://github.com/user-attachments/assets/f0db51ef-e273-491c-9736-a668a38e991e">
</div>


---
<div align="center">
  ➡️<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview">Competition Website Here</a>⬅️
</div>  

---


<h1>Intro</h1>

**This Repo contains the code and the result of our participation in an indefinite-period Kaggle Project**🏘️📈

###### Project:

- Goal: Use (advanced) regression algorithm to predict the house price. Train the Regression model on training data, and predict the price of each house in test data.
- Training Data: A set of 1460 house infos from Ames Housing dataset , consisting of 79 features (e.g.: House Area, Number of Neighbours, Built Year, etc.) and 1 _Sale Price_ as the label
- Test Data: A set of 1460 house infos, same features, **Unlabelled** (Labels stored on Kaggle and will be evaluated upon submission).
- Evaluation Metric:  Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.
- Competition Time: Indefinite, maximum 10 submissions/day.

###### Our team:

- Team name: Coconut Coffee (Because our idea and journey began from a Café in Amsterdam, with a cup of freshly-made coconut coffee! 🥥)
- Contributors: <a href="https://github.com/BruceLeo99">Yixing</a>, <a href="https://github.com/PT-VU">PT</a>
- Rank: 2715
- **Best RMSE score: 0.15644**
- Last Submission: 05 Sep 2024

  <div align="center">
    <img width="923" src="https://github.com/user-attachments/assets/a556dfab-3987-4dce-9f5c-3a675352ce2b">
  </div>


# Techniques we used

#### Exploratory Data Analysis (EDA):
Before starting implementing the algorithm, we spent some time to explore and visualize the data. <br>
We carefully viewed:

- The datatype of each feature
- Density of Missing Values
- Range of Sales Price
- Distributions of each feature and Sales Price
- Outliers of each feature

#### Feature Engineering
After carefully viewing the data, we made decisions on handling the data by:

1. **Fixing Datatype:** Some categorical data (e.g.: Quality label from 1 to 10 and Material Quality from 1-5) were typed as Integers, which might "confuse" the algorithms. We changed these Datatypes to Categorycal (Object in Pandas DataFrame) to ensure that they won't be treated as real values.
2. Removing outliers: Several outlier removal methods were applied:
  - **Remove by 3 IQR**
  - **Remove by 5 IQR**
  - **Winsorization**: remove _x_ lower percentile and _y_ upper percentile
    - During the training, we realized that **removing by 3 and 5 IQR caused significant data loss (nearly 60 percent!) and resulted poor model performance, hence we applied Winsorization to preserve the data integrity.**
    - Bounds were adjusted by _Experimenting_ and _observation of the outlier range_.
4. Imputing Missing Values:
  - Numerical Data: **Forward fill, Backward fill, Iterative Imputing**
  - Categorical Data (e.g.: Street Type, Alley Type, Swimming Pool Type): We treated Categorical Data as 2 types: **Predictable and Non-Predictable**. A Categorical Data is predictable if it does not have too many missing values and can be predicted from the data of non-missing individuals; non-predictable if it has lots of missing values.
    - Predictable Categorical Data: **Use the same imputing technique as Numerical Data**
    - Non-predictable: Treat missing values as a separate value, namely "No Value", and fill it in. For example, there were only 7 houses with a swimming pool, and it is not sensible to "assume" that other houses have a swimming pool as well, therefore, we filled in the missing values of swimming pool type as "No Swimming Pool".
4. Data Transforming: Normamizing the data with different scales of each feature: **min-max, "absolut-max, z-score", log,  yeo-johnson,  "squareroot**
   
#### Algorithms

###### Baseline and dummy model

- Randomly-generated "fake" Sales Price within the range of the Sales Price in training data
- LASSO Regression Model on **unprocessed data**, $\alpha \in [0.0001, 1]$, 5-fold CV, repeat 5 times with train-validation splitting randomness control (_random_state=1_) to ensure that there is no duplication
  - Initially evaluate the RMSE (before submitting to Kaggle and being evaluated by the Kaggle System) of each $\alpha$.

###### Experimental Models:

**Train LASSO and Elastic-net Model with multiple types of hyperparameters, and training data varying in their outlier removal methods, trainsformation and imputing techniques**. <br><br>

**Hyperarameters:**
- alpha: $\alpha \in [0.0001, 100]$
- l1 ratio (for Elastic-net only): $l1 \ ratio \in [0,1]$
  
For both LASSO and Elastic-net if evaluated by Grid Search CV:
- max_iter: $max\\_iter \in \[1000,10000]$, numver of iterations to compute maximally
- tolerance: 0.0001, 0.01, 0.1, 0.5, threshold difference in Model's covariances of each iteration that is allowed. Stop training when the covariance is smaller than the threshold to prevent overfitting and wasting of computational power.

#### Evaluation:

We firstly used one of the Validation Methods (5-fold CV with 5 repeats or Grid Search CV) to find out the best-reported hyperparameter combination, then used these hyperparameters to run an _initial test_ (random 70-30 train-test split) and score the RMSE, then visualized the **residual plot** to ensure the result is _Homoscedastic_ (no multicoliniearity, which might cause poor model performance)

# Results





