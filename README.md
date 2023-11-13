## Cybersecurity Machine Learning, threat detection sniffing packets on the network
This project is an initiative to sniff packets on the network, processes the data using ML and classify the traffic as normal or malicious.

## Project Status: [In Progress]

### Project Intro/Objective
Improve resource utilization, time and effort using targeted marketing. The purpose of this project is to use machine learning techniques to identify customers who are likely 
to pay attention to the marketing campaign and consider or invest in term deposits with the bank. Idea is to improve the 1% success rates to much larger %'s so 99% wasted time and resource can be leveraged optimally targeting the right customers.

### Methods Used
* Statistics
* Supervised Machine Learning
* Data Visualization
* Classification Modeling
* Optimization techniques

### Technologies
* Python
* ML models: 
    - Logistic Regression, 
    - KNN, Decision Trees, 
    - SupportVectorMachine(SupportVectorClassifier) 
    - & RandomForestClassifier 
* Jupiter notebook
* sklearn, pandas, numpi, time

## Project Description
Bank has collected 21 features or columns of around 41188 rows or records. These are statistics and data about the clients. Marketing term deposit indiscriminately to all these clients historically has given a success rate of 1%. Not just a low success rate but has given bad customer experience. This initiative is to statistically study the clients, category of clients, observe their pattern, past success on term deposits, financial potential and come up with target clients to market the product (term deposits). The idea is to spend efficient resources, time, effort and cater to better customer experience.

## Needs of this project
- ML modeling skills, python
- data exploration/descriptive statistics
- data processing/cleaning
- statistical classification modeling
- writeup/reporting

## Getting Started and phased execution, processes
1. Clone this repo (UC Irvine ML Repo [Detailed Dataset to Download](https://archive.ics.uci.edu/dataset/222/bank+marketing)).
2. Understand the business challenges, use case, problem to solve, features used in the dataset, co-relation understanding with target and among features 
3.EDA(Exploratory Data Analysis)
    * Analyze for null values, undefined values, quantity of each, normalized baseline target outcome
    * Drop features that are of minimal to no use to analyze the target column. Balance tradeoffs between data imputng vs dropping based on quantity missing
    * Identify what features are categorical, ordinal and scalar.
    * Reduce features to the absolute minimum and essential features dataset.
    * Prepare features using column transformation of the dataset based on feature identification.
    * treating data - Feature Columns transformation (onehot encoding, label encoding, passthroguh or standardscalar)
3. Once Data transformation has been completed analyze the number of features that we end up with. In this case its large. (55 to be precise)
4. Since the dataset has too many features and also as a best practice analyze the dataset with corelation matrix. Preferably setting thresholds and drop features that are highly corelated in this case (>.7) is being used. Dropping 7 features ending up with 48 features to analyze. 
5. While this may be good to fit classification models it is still too large for model hyperparameter tweaks and finetuning. 
6. Adopt techniques to further strip down just the bare essentil features to tackle or identify this classification outcome. RandomForest is being used in this case to reduce the features to 5 to bring it down to managable features for classification.
7. Test out all major classification models benchmarking fit times, accuracy and interpretability.
8. Decide on the best classifier model, tweak it for better accuracy.
9. Eventually implement / train the model with best hyper parameters to get accurate desired outcome.
10. Our desired outcome in this case is For the given customer the model should give an output of "1" for potential customer to target or "0" not a customer to target for term deposit clientel. Using this bank can decide to either place a call or refrain from marketing term deposit product to the customer.

## Featured Notebooks/Analysis/Deliverables
* [Python Code in Jupiter Notebook](https://github.com/aaghouse/bank_marketing_campaign/blob/master/bank_mktng_classification_y-or-n_term_deposit.ipynb)
* [Data Set](https://github.com/aaghouse/bank_marketing_campaign/tree/master/bank%2Bmarketing/bank-additional/bank-additional)
* [Data feature description1](https://github.com/aaghouse/bank_marketing_campaign/blob/master/images/DataSetFeatures1of2.png)
* [Data feature description2](https://github.com/aaghouse/bank_marketing_campaign/blob/master/images/DataSetFeatures2of2.png)
* [Results Summary](https://github.com/aaghouse/bank_marketing_campaign/blob/master/ResultsSummary.md)
* [Reference Paper](https://github.com/aaghouse/bank_marketing_campaign/blob/master/CRISP-DM-BANK.pdf)

## Results
Business context, when the bank contacted its target clients or so called target clients majority of the clients declined to enroll in term based deposits. Analyzing the given data set 88.73% declined to enroll.  

In DataScience language, baseline accuracy of the majority outcome "0" of the given data set is 88.73%
Post DataScience techniques baed on the quantitative statistical analysis and modeling we can target clients with > 90%
accuracy that they would enroll in term based deposit product.

![Classification Models with accuracy, fit times & interpretability](Output_All_models-Final-Results-Plots.png)

Results Summary (Detailed results analysis [Final Summary](https://github.com/aaghouse/bank_marketing_campaign/blob/master/ResultsSummary.md))
## Contact 
* Author: Abdul Ghouse, email: aghouse@gmail.com
