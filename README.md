# **House Prices - Advanced Regression Techniques**

This Repository is my submit prediction in [Kaggle Competition - Houses Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

* The description features is in the file _"data_description.txt"_, the data of training and testing is in the files _train.csv_ and _test.csv_.

* The models was training for each techniques, is save in _"models_sav"_ fold.    

This competition is very important for me. I'm a Data Science and Machine Learning enthusiast, I have been to study these all themes for one year. Thus, these competitions of Kaggle are good opportunities for practice all techniques that I was a study of theoric form. And as are evident, I was take as reference many notebooks, when the authors to dominate the area, offer all knowledge of years and a useful guide for my studies. 

Somes of this authors is:
1. [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) by Serigne. He has excellent work in Stacked Regression, but to a great work in Data Cleaning and Data Preprocessing that is very useful for this notebook.
2. [Regression & Classification with Ames Housing Data](https://www.perkinsml.me/ames-housing) by Matthew Perkins. His EDA is beautiful and it served for a preliminary analysis. Check this portfolio. ;)

All phases it is contained in this notebook is of the biggest common:
1. Data Processing and Analysis.
* Outliers - identify the values in SalesPrice that can be noisy and will affect of train techniques of regression evaluated.
* Normalize of Target Variable - normalize SalesPrice with log function and we can get a better probability.
* Features Engineering - data cleaning, imputing, engineering variables, Encondig, BOX - COX transformation  and Dummy Variables

2. Modeling
* Random Forest Regression - prediction SalesPrice with classical forest regression with Scikit-Learn
* Gradient Boosting Regression - with gradient boosting tree techniques, cross-validation and grid search to find the best model with optimal hyperparameters.
* Linear Regression Techniques - With some techniques linear regression with penalty as LASSO and Bayesian Ridge.

3. Selection of the Best Model.
* Submission of results
* 

## Note for futures branch or revision

This repository in fold Notebooks for now, only is _"houses_prices_script.py"_ in repository. In the next branches, we will adding __jupyter notebook__ format which will be the same that notebooks in Kaggle.

