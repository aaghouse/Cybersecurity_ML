## Cybersecurity Machine Learning, threat detection sniffing packets on the network & classifying type of threat
This project is an initiative to processes network packet headers or information using ML and classify the traffic as normal or malicious. On successful classification we then identify threats and bin it according to the type of threat using neural networks..

## Project Status: [Completed]

### Project Intro/Objective
The project has two parts, part I is to classify network traffic as malicious or normal and part II is to identify the type of threat. We use data captured from a network packet sniffer that has malicious traffic and normal traffic in the ratios of 35% and 65%. 
* **(Part I)** Train various classification models to identify threats. Once trained we evaluate models and capture ideal hyper parameters among various models. We then picked the top model to classify network traffic with high accuracy to identify threats. Our goal is to be able to do dynamic classification that can be followed up to raise alerts.
* **(Part II)** Once we have identified traffic to be normal or malicious we then identify malicious traffic type using neural networks. We evaluate various neural networks, identify the best one for our use case and train them further to improve accuracy.
  - Prospects as next step, output from this neural network could be leveraged to dynamically produce firewall rules and improve organizations security posture instantaneously.


### Methods Used
* Statistics
* Supervised Machine Learning
* Data Visualization
* Classification Modeling
* Optimization techniques

### Technologies
* Python
* ML models 
    - Logistic Regression 
    - KNN, Decision Trees 
    - & RandomForestClassifier 
* Neural Networks
    - ANN (Artificial Neural Networks), DNN (Deep Neural Networks)
    - CNN (Convolutional Neural Networks)
    - RNN (Recurrent Neural Networks)
* Jupiter notebook
* sklearn, pandas, numpi, time, ConfusionMatrixDisplay, RocCurveDisplay

## Project Description
Data collected has 461,043 records. We use this data to train the model to learn and identify threats. Once trained we perform binary classification to identify threats. Once threat has been identified we then identify the type of threat using multi class classification.

## Needs of this project
- ML modeling skills, python
- data exploration/descriptive statistics
- data processing/cleaning
- statistical classification modeling (binary and multi class), fine tuning models
- neural networks
- writeup/reporting

## Getting Started and phased execution, processes
1.	Clone this repo (UC Irvine ML Repo Detailed Dataset to Download).
2.	Understand the business challenges, use case, problem to solve, features used in the dataset, co-relation understanding with target and among features 
3.	* EDA(Exploratory Data Analysis)
        - Analyze for null values, undefined values, quantity of each, normalized baseline target outcome
        - Drop features that are of minimal to no use to analyze the target column. Balance tradeoffs between data imputng vs dropping based on quantity missing. In this case we were able to reduce the features from 45 to just 15.
        - Identify what features are categorical, ordinal and scalar.
        - Reduce features to the absolute minimum and essential features dataset.
        - Prepare features using column transformation of the dataset based on feature identification.
        - Treating data - Feature Columns transformation (onehot encoding, label encoding, passthrough or standardscalar)
4.	Once Data transformation has been completed analyze the number of features that we end up with. In this case it’s large. (49 to be precise, encoding seems to have rehydrated the feature sets)
5.	Since the dataset has too many features and also as a best practice analyze the dataset with corelation matrix. Preferably setting thresholds and drop features that are highly corelated in this case (>.7) is being used. Dropping 3 features ending up with 46 features to analyze.
6.	While this may be good to fit classification models it is still too large for model hyperparameter tweaks and finetuning.
7.	Adopt techniques to further strip down just the bare essential features to tackle or identify this classification outcome. RandomForest is being used in this case to reduce the features to 24 to bring it down to manageable features for classification. This is still considered large feature set.
8.	Test out all major classification models benchmarking fit times, accuracy, and interpretability.
9.	Decide on the best classifier model, tweak it for better accuracy.
10.	Eventually implement / train the model with best hyper parameters to get accurate desired outcome.
11.	Our desired outcome in this case is for the given network packets record to be identified as "1" for threat or identify it as "0" for normal traffic. Using this we can then decide to alert, further classify into various threat histograms or even develop dynamic firewall rules to prevent any attack traffic from entering the network.


## Featured Notebooks/Data/Model cards/Analysis/Deliverables
#### Python Code in Jupiter Notebook
- **Part I**
  - [File 1 of 4, binary classification using ML models evaluation, cyber_ml_capstone.ipynb](https://github.com/aaghouse/Cybersecurity_ML/blob/main/cyber_ml_capstone.ipynb)
  - [File 2 of 4, Optimized model, optimized_model.ipynb](https://github.com/aaghouse/Cybersecurity_ML/blob/main/optimized_model.ipynb)
  - [File 3 of 4, Cyberthreat geo location, cool_plots.ipynb](https://github.com/aaghouse/Cybersecurity_ML/blob/main/cool_plots.ipynb)
- **Part II**
  - [File 4 of 4, multi class classification using neural networks Identifying threat type, neural_net_multi_class_classification.ipynb](https://github.com/aaghouse/Cybersecurity_ML/blob/main/neural_net_multi_class_classification.ipynb)
#### Data
- **Data Source**
  - [Data Set](https://github.com/aaghouse/Cybersecurity_ML/tree/main/dataset)
- **Data Description**
  - [Data sheet](https://github.com/aaghouse/Cybersecurity_ML/blob/main/dataset/cyber_sec_network_datasheet.pdf)
#### Model Information
- **Model Cards**
  - [model card1]()
  - [model card2](https://github.com/aaghouse/Cybersecurity_ML/blob/main/dataset/MulticlassClassificationModelCard-2.pdf.pdf)
#### Reference Paper
* [Reference Paper on Data Collection](https://github.com/aaghouse/Cybersecurity_ML/blob/main/dataset/Testbed%20%26%20attacks%20of%20TON_IoT%20datasets.pdf)

## Results
![project-flow](https://github.com/aaghouse/Cybersecurity_ML/assets/90729963/a5aee237-db5b-4888-8a93-d31a192dc97d)

Cyber threat detection and deterrence are a regular part of corporate operations. Speed at with we can detect malicious traffic, categorize it, neutralize, and prevent it is critical. As a part of this project, we fed in data from network traffic packet captures, leveraged machine learning models, evaluated various models, picked the most optimal model, fine tuned it and were able to classify network traffic as malicious or not with high degree of accuracy (> 98%) (Part I). Once we identified the threat, we further utilized neural network, evaluated various networks, identified most optimal network and were able to classify the threats (9 types of threats) with (~78%) accuracy and minimum loss(<0.4). 
The model is now ready to be tried and utilized for production. (part I) As a part of the next step this output can be leveraged to raise alerts to deter cyberthreat and the (part II) output of neural network can be utilized to develop dynamic firewall rules to deter further threats and fortify cyber defense.

[Results final summary](https://github.com/aaghouse/Cybersecurity_ML/blob/main/results-summary.pdf)
## Contact 
* Author: Abdul Ghouse, email: aghouse@gmail.com
