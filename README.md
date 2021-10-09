This repository contains data science Capstone projects that I did while studying at Turing College.


## Home Credit Default Risk 


<img src="https://raw.githubusercontent.com/GQ21/Data-Science-Projects/main/Credit_Risk/img_homecredit.png" align="centre">

I choosed to participate in [Home Credit Default Risk kaggle competition](https://www.kaggle.com/c/home-credit-default-risk/overview) where I was challenged to make a model which would ensure that home credit clients capable of repayment wouldn't be rejected and the ones that potentialy default would be rejected. Main challenge was to analyze, clean, aggregate and merge provided Big Data - 7 dataframes which took over 2.5 GB of space. I explored multiple models and tested their speed. I experimented with different imputing, outliers detection, feature selection algorithms. I set a goal to get at least median private leaderboard score and I succeded. I found that feature engineering was one of the leading factor in achieving high kaggle score. 

[Take a look!](https://github.com/GQ21/Data-Science-Projects/blob/main/Credit_Risk/8_final_merge.ipynb)

### Tools
*   Python 3.6 
*   Jupyter Notebook 
*   Pandas
*   Scikit-Learn
*   Lightgbm
*   Matplotlib
*   Seaborn
*   Bayes_opt


## Fatal Police Shootings

<img src="https://github.com/GQ21/Data-Science-Projects/blob/main/Police_Shootings/img_shootings.jpg" align="centre">


[Police brutality in the United States](https://en.wikipedia.org/wiki/Police_brutality_in_the_United_States) has been an nationwide issue since the 20th century. Public safety of U.S. citizens is a typical argument to justify the controversially high number of fatal shootings.

I took [kaggle data](https://www.kaggle.com/washingtonpost/police-shootings)  and decided to analyze list of issues that follow police brutality. I thought about  multiple ideas on how to tackle these found issues. I used hypothesis testing to check if there were any difference between years shooting records. I tested clustering techniques to find if there were any patterns that could signal racial descrimination. Unfortunately used clustering algorithms wasn't working with categorical data therefore I decided to make a simple logistic regression and see if there are specific circumstances were people with certain race can be shooted a bit more likely than other races.

[Take a look!](https://github.com/GQ21/Data-Science-Projects/blob/main/Police_Shootings/Police_Schootings.ipynb)


### Tools
*   Python 3.6 
*   Jupyter Notebook 
*   Pandas
*   Scikit-Learn
*   Matplotlib
*   Seaborn
*   Plotly
*   Scipy
*   Kneed
