# [Python] ML_Churn_User_Prediction

## I. Introduction
### 1. Business question
One ecommerce company has a project on predicting churned users in order to offer potential promotions. The company wants to answer questions below:
1. What are the patterns/behavior of churned users ? What are your suggestions to the company to reduce churned users.
2. Build the Machine Learning model for predicting churned users.

### 2. Dataset
The dataset records the usage characteristics of 5630 users. Each user is labeled as churn or not.
Dataset include these following main fields:

| Variable                      | Description                                                   |
|-------------------------------|---------------------------------------------------------------|
| CustomerID                    | Unique customer ID                                            |
| Churn                         | Churn Flag                                                    |
| Tenure                        | Tenure of customer in organization                            |
| PreferredLoginDevice          | Preferred login device of customer                            |
| CityTier                      | City tier                                                     |
| WarehouseToHome               | Distance in between warehouse to home of customer             |
| PreferredPaymentMode          | Preferred payment method of customer                          |
| Gender                        | Gender of customer                                            |
| HourSpendOnApp                | Number of hours spend on mobile application or website        |
| NumberOfDeviceRegistered      | Total number of deceives is registered on particular customer |
| PreferedOrderCat              | Preferred order category of customer in last month            |
| SatisfactionScore             | Satisfactory score of customer on service                     |
| MaritalStatus                 | Marital status of customer                                    |
| NumberOfAddress               | Total number of added added on particular customer            |
| Complain                      | Any complaint has been raised in last month                   |
| OrderAmountHikeFromlastYear   | Percentage increases in order from last year                  |
| CouponUsed                    | Total number of coupon has been used in last month            |
| OrderCount                    | Total number of orders has been places in last month          |
| DaySinceLastOrder             | Day Since last order by customer                              |

### 3. Method
Supervised learning with Scikit-learn on Python
- Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.
- As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process.
- Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

## II. Proccess (Chua sua xong)
### 1. Cleaning and transforming dataset
- The maximum number of null values in each column is low: 137/200000 values => Remove null values from dataset
- Values in 'user_id' column was changed from original data for security, so some duplicated values appear => Remove those duplicate values
- Original data is organized as wide form, each field is devided into 10 columns to show infomation of 10 months => Convert dataframe form wide to long form for better manipulation. Then calculate the mean, mode of each column group by 'user_id' to combine 10 months into a single row

**Dataset after transforming**

![image](https://github.com/thuhuongphan11/Python_Cohort_Analysis/assets/141643891/7f001184-3913-4d25-8898-5ad6ff100b40)
### 2. EDA and select features of the model
#### EDA
![image](https://github.com/thuhuongphan11/Python_Cohort_Analysis/assets/141643891/11ea5be8-1ba8-4c81-8922-de397a98db39)

- The heatmap shows that 'HA_TANG_', 'THIET_BI_', 'NOD_PSLL_DATA_', 'SO_NGAY_SU_DUNG_' have a high correlation with the target column 'thuc_4g_'. But 'HA_TANG_' and 'THIET_BI_' are correlated, -> we choose 1 of these 2 features.   
  => Choose 'HA_TANG_', 'NOD_PSLL_DATA_' and 'SO_NGAY_SU_DUNG_' to apply to the model.
- There are some columns that we think may affect the target "thuc_4g" ('NOD_PSLL_THOAI_', 'TUOI_KH_', 'IS_DCOM_') => EDA to check the relation with the "thuc_4g" column.
#### Features selection
After EDA we will keep the below columns to the model:
- 'HA_TANG_': The infrastructure used by customer. 2G-3G is an old technology, 4G is a new technology in the future
- 'NOD_PSLL_DATA_': Number of days using data in that month
- 'NOD_PSLL_THOAI_': Number of days using voice calls in that month
- 'SO_NGAY_SU_DUNG_': Number of days using one of their services: Voice, Data, SMS
- 'thuc_4g_': target column
## III. Model training and evaluation
- Encoding & Normalizing dataset
- Apply to models: Logistic Regression, K Nearest Neighbors (KNN), Decision Tree and Random Forest

## IV. Conclussion
Comparing the balance accuracy of 4 models, we can see that Logistic Regression has the highest test set's F1-score (0.929). Logistic Regression also has the least difference between the F1-score of the test set and the train set (0.929 and 0.93, respectively).
=> Choose **Logistic Regression** as the final model used to predict 4G service customers for this corporation.

***Confusion matrix of Logistic Regression model***

![image](https://github.com/thuhuongphan11/Python_Cohort_Analysis/assets/141643891/de737434-9ba1-4bc2-8bb2-f62028c9f1a2)
