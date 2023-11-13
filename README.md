## Cybersecurity Machine Learning, threat detection sniffing packets on the network
This project is an initiative to sniff packets on the network, processes the data using ML and classify the traffic as normal or malicious.

## Project Status: [In Progress]

### Project Intro/Objective
We use data captured from a network packet sniffers that has malicious traffic and normal traffic in the ratios of 35% and 65%. Train various classification models to identify threts. Once trained we evaluate models and capture idel hyper paramaters among various models. We then pick the 
top model to classify network traffic with high accuracy to identify threts. Goal is to be able to do dynamic classification that can be followed up to raise alerts or even potentially come up with histogram of the type of threts and develop dynamic firewall rule sets to block further attacks.

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
    - & RandomForestClassifier 
* Jupiter notebook
* sklearn, pandas, numpi, time, ConfusionMatrixDisplay, RocCurveDisplay

## Project Description
Data collected has 461,043 records. We use this data to train the model to learn and identify threats. Once trained we perform binary classification
to identify threats. 

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
3. Once Data transformation has been completed analyze the number of features that we end up with. In this case its large. (49 to be precise)
4. Since the dataset has too many features and also as a best practice analyze the dataset with corelation matrix. Preferably setting thresholds and drop features that are highly corelated in this case (>.7) is being used. Dropping 3 features ending up with 46 features to analyze. 
5. While this may be good to fit classification models it is still too large for model hyperparameter tweaks and finetuning. 
6. Adopt techniques to further strip down just the bare essentil features to tackle or identify this classification outcome. RandomForest is being used in this case to reduce the features to 24 to bring it down to managable features for classification. This is still considered large feture set.
7. Test out all major classification models benchmarking fit times, accuracy and interpretability.
8. Decide on the best classifier model, tweak it for better accuracy.
9. Eventually implement / train the model with best hyper parameters to get accurate desired outcome.
10. Our desired outcome in this case is For the given network packets record to be identified as "1" for threat or identify it as "0" for normal traffic. Using this we can then decide to alert, further classify into various theat histograms or even develop dynamic firewall rules to prevent any attack traffic from entering the network.

## Featured Notebooks/Analysis/Deliverables
* [Python Code in Jupiter Notebook]()
* [Data Set]()
* [Data sheet](https://github.com/aaghouse/bank_marketing_campaign/blob/master/images/DataSetFeatures1of2.png)
* [model card](https://github.com/aaghouse/bank_marketing_campaign/blob/master/images/DataSetFeatures2of2.png)
* [Results Summary]()
* [Reference Paper on Data Collection]()

## Results
TBD

![Classification Models with accuracy, fit times & interpretability](Output_All_models-Final-Results-Plots.png??)

Results Summary (Detailed results analysis [Final Summary](https://github.com/??))
## Contact 
* Author: Abdul Ghouse, email: aghouse@gmail.com
