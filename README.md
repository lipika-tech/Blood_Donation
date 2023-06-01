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




