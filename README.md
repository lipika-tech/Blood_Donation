# Blood_Donation
# ABSTRACT
"Blood is the most precious gift that anyone can give to another person — the gift of life." ~ 
World Health Organization
Forecasting blood supply is a serious and recurrent problem for blood collection managers: in 
January 2019, "Nationwide, the Red Cross saw 27,000 fewer blood donations over the holidays 
than they see at other times of the year." Machine learning can be used to learn the patterns in 
the data to help to predict future blood donations and therefore save more lives.
In this Project, you will work with data collected from the donor database of Blood Transfusion 
Service Center in Hsin-Chu City in Taiwan. The center passes its blood transfusion service bus 
to one university in Hsin-Chu City to gather blood donated about every three months. The 
dataset, obtained from the UCI Machine Learning Repository, consists of a random sample of 
748 donors. Your task will be to predict if a blood donor will donate within a given time 
window. You will look at the full model-building process: from inspecting the dataset to using 
the tpot library to automate your Machine Learning pipeline.
To complete this Project, you need to know some Python, pandas, and logistic regression. We 
recommend one is familiar with the content in DataCamp's Manipulating DataFrames with 
pandas, Preprocessing for Machine Learning in Python, and Foundations of Predictive 
Analytics in Python courses.

# About the Project
 I believe donating blood is important. Good data-driven systems for tracking and 
predicting donations and supply needs can improve the entire supply chain, making sure that 
more patients get the blood transfusions they need.
 In the United States, the American Red Cross is a good resource for information about 
donating blood. According to their website:
 Every two seconds someone in the U.S. needs blood.
 More than 41,000 blood donations are needed every day.
 A total of 30 million blood components are transfused each year in the U.S.
 The blood used in an emergency is already on the shelves before the event occurs.
 Sickle cell disease affects more than 70,000 people in the U.S. About 1,000 babies are 
born with the disease each year. Sickle cell patients can require frequent blood 
transfusions throughout their lives.
Amazingly, only around 5% of the eligible donor population actually donate (Linden, Gregorio 
et al. 1988, Katsaliaki 2008). This low percentage highlights the risk humans are faced with 
today as blood and blood products are forecasted to increase year-on-year. This is likely why 
so many researchers continue to try to understand the social and behavioral drivers for why 
people donate to begin with. The primary way to satisfy demand is to have regularly 
occurring donations from healthy volunteers.

# Understanding the Data set
The data file blood_donation.csv contains the information used to create the 
model. It consists of 748 rows and five columns. The columns represent the 
variables, and the rows represent the instances.
The number of input variables, or attributes for each sample, is 5. All input 
variables are numeric -valued and represent features from blood donors. 
The target variable is donation, being 0 no blood donation and 1 blood donation 
for the last campaign. The following list summarizes the variables information:
The next list describes the variables information:
 Recency: Months since the last donation.
 Frequency: Total number of donations.
 Monetary: Total blood donated.
 Time: Months since the first donation.
 Whether he/she donated blood in March 2007 : True if the person donated 
in the last campaign, false otherwise.
Finally, the use of all instances is selected. Each patient has an instance that 
contains the input and target variables.
We'll now use train_test_split() method to split transfusion DataFrame.
Target incidence informed us that in our dataset 0s appear 76% of the time. We want to keep 
the same structure in train and test datasets, i.e., both datasets must have 0 target incidence of 
76%. This is very easy to do using the train_test_split() method from the scikit learn library -
all we need to do is specify the stratify parameter. In our case, we'll stratify on 
the target column.


 More than 1.6 million people were diagnosed with cancer last year. Many of them will 
need blood, sometimes daily, during their chemotherapy treatment.
 A single car accident victim can require as many as 100 pints of blood
One of the interesting aspects about blood is that it is not a typical commodity. First, there is 
the perishable nature of blood. Grocery stores face the dilemma of perishable products such as 
milk, which can be challenging to predict accurately so as to not lose sales due to expiration. 
Blood has a shelf life of approximately 42 days according to the American Red Cross 
(Darwiche, Feuilloy et al. 2010). However, what makes this problem more challenging than 
milk is the stochastic behavior of blood supply to the system as compared to the more 
deterministic nature of milk supply. Whole blood is often split into platelets, red blood cells, 
and plasma, each having their own storage requirements and shelf life. For example, platelets 
must be stored around 22 degrees Celsius, while red blood cells 4 degree Celsius, and plasma 
at -25 degrees Celsius. Moreover, platelets can often be stored for at most 5 days, red blood 
cells up to 42 days, and plasma up to a calendar year.


# METHODOLOGY
The biomedical signal analysis of Blood and PPH requires several steps, including data preprocessing to prepare the data, selecting appropriate features using feature extraction and 
engineering, model selection and validation, and tuning of hyperparameters. All these steps 
aim to effectively produce an accurate machine learning model to estimate Blood from PPH. 
To achieve this goal, a tree-based pipeline optimization tool (TPOT) is used in this paper to 
estimate the blood loss from PPH. To simplify, TPOT uses genetic programming from the 
Python package DEAP to pick a series of pre-processing data functions and ML classification 
or regression algorithms to optimize the model’s performance for a dataset of interest. In 
addition to the ML algorithm, the TPOT model pipeline, as presented in the example illustrated 
in Fig, includes a variety of data transformers implemented in the Scikit-learn Python library, 
such as various pre-processors (Min-Max Scaler, Standard Scaler, Max Abs Scaler, 
Normalizer, polynomial features expansion) and feature selectors (Select Percentile, Variance 
Threshold, recursive feature elimination). In some instances, designing a new feature set may 
help extract valuable information (e.g., when a selected method analyses one feature 
simultaneously while complex feature interactions are present in the dataset). TPOT also 
provides several custom function constructor implementations: Zero counts (count of zero/nonzeros per sample), stacking estimator (SE) (generates predictions and class probabilities with a 
classifier of choice as new features), one hot encoder (converts categorical features to binary 
features), and a range of sklearn transformer implementations: PCA, independent component 
analysis, and a selection of sklearn transformer implementations (Nystroem, RBF Sampler). 
The complete TPOT configuration includes 11 classification algorithms, 14 feature 
transformers, and five feature selectors. TPOT uses a tree-based structure to incorporate all 
these operators (Fig). Any pipeline begins at the root of the tree structure with one or more 
copies of the entire dataset and continues with the feature transformation/selection operators 
mentioned above or with the ML algorithm. The original dataset is then modified and moved 
down the tree to the next operator or if there are several copies of the dataset, they can be 
merged into a single set using a combination operator.

![image](https://github.com/lipika-tech/Blood_Donation/assets/76075950/6628f525-3b7d-4131-8332-f0676da0b41e)


TPOT designs a generic algorithm for searching a wide range of supervised classification 
algorithms that adopt the Python Scitkit learning library, including preprocessors, transformers, 
feature selection techniques, estimators, and their hyperparameters, without any domain 
knowledge or human data inputs. As demonstrated in the correlation matrix heatmap for the 
dataset of this study, the stronger correlation on both ends of the spectrum presented in a darker 
color and weaker correlation has been presented in a lighter shade color. Thus, the feature 
extraction pipeline must precisely map the PPG diastolic peak to the foot (signal minimum) of 
the BP signal. In the following sub-sections, the details of TPOT phases will be elaborated.
TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines 
using genetic programming.


use case diagram
![image](https://github.com/lipika-tech/Blood_Donation/assets/76075950/9a1109fb-062f-4586-b708-d0a8029e4dd9)



TPOT will automatically explore hundreds of possible pipelines to find the best one for our dataset. Note, the 
outcome of this search will be a scikit-learn pipeline, meaning it will include any pre-processing steps as well 
as the model.
We are using TPOT to help us zero in on one model that we can then explore and optimize further

# Language and Platform Used
2.2.1 Language: Python
Python is a high-level, general-purpose programming language. Its design 
philosophy emphasizes code readability with the use of significant indentation via 
the off-side rule. Python is dynamically typed and garbage-collected.
It can be used to 
 AI and machine learning
 Data analytics 
 Data Visualization
 Programming applications

2.2.2 IDE: JupyterLab
JupyterLab is the latest web-based interactive development environment for 
notebooks, code, and data. Its flexible interface allows users to configure and 
arrange workflows in data science, scientific computing, computational journalism, 
and machine learning. A modular design invites extensions to expand and enrich 
functionality.
 Easy to convert: Jupyter Notebook allows users to convert the notebooks 
into other formats such as HTML and PDF.
 It also uses online tools and nbviewer which allows you to render a publicly 
available notebook in the browser directly.
 Live code 
 Equations 
 Computational output
 Visualizations
 Multimedia resources
 Explanatory text.

# IMPLEMENTATION
3.1 Gathering Requirements and Defining Problem Statement
This is the first step wherein the requirements are collected from the clients to understand the
deliverables and goals to be achieved after which a problem statement is defined which has to be 
adhered to while development of the project.
3.2 Data Collection and Importing
Data collection is a systematic approach for gathering and measuring information from a variety 
of sources in order to obtain a complete and accurate picture of an interest area. It helps an 
individual or organization to address specific questions, determine outcomes and forecast future
probabilities and patterns.
Our dataset is from a mobile blood donation vehicle in Taiwan. The Blood Transfusion Service Center drives 
to different universities and collects blood as part of a blood drive. We want to predict whether or not a donor 
will give blood the next time the vehicle comes to campus.
The data is stored in datasets/transfusion.data and it is structured according to RFMTC marketing 
model (a variation of RFM).

Jupyter notebook attached

# Conclusion
The aim of this Blood Donation Application is to improve the communication 
with the people who are in need of blood and the persons who are willing to 
donate blood. This will reduce the barrier between blood donors and the people 
in sever need of blood.
So our research’s objective is to build a community of blood donor and to make 
sure that we can come forward to donate blood as it can make sure the return of 
a dying man again into the light of life.
Donating blood and blood components are easier than ever. Connecting blood 
donors and needy reduces time which increases the possibility of saving lives 
and also eliminates the shortage of blood. It is one of its first and only unique 
applications available with feature of realtime map and machine learning 
algorithm for finding the best suitable donor. It uses the internet connection to 
let us search blood donors and recipient. This enables to find the best matches 
among the donors available with the help of machine learning algorithms. The 
algorithms are capable of analyzing the profile of each donor and find the best 
fit ones with respect to health condition and lifestyle. Moreover, It is also 
capable of showing the exact number of the donors in the map who are willing 
to donate blood. This will make the easiest and fastest way to get a best match 
blood donor.


# FUTURE SCOPE:
The work outputs the power to save lives in the palm of your hand. Donating blood and platelets 
is easier than ever. Find nearby Red Cross blood drives, schedule appointments, earn rewards 
from premier retailers, and follow your blood’s journey from donation through delivery (when 
possible), and create or join a lifesaving team and track its impact on a national leader board. 
The future plan is this: 
1. Send Push Notification to the persons who are selected and then if the person accept the 
request another notification will be sent to the sender and connection will be established.
2. There will be a request and accept button for sending and receiving push notification.
3. All the nearby Hospitals will be shown in the app 
4.The whole blood donation profile will be saved in the database.





