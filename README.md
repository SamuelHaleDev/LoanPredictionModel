# LoanPredictionModel
This model is based off of a Kaggle loan data set. https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset. I went for a standard linear layer for this one as it gave me the best results. The model at its best got ~75% accuracy. Understandably since there was not too much data to train with. Some of the data didn't make since even to me seeing two identical candidates with something as simple as just different geographical residency and that being the difference between getting and not getting a loan. 

As for how to run this it requires a few libraries that you can install through PyCharm in the virtual environments it offers.
1) torch
2) torchvision
3) pandas
4) matplotlib
5) numpy

You need to download the dataset and in the code specify the path of the dataset locally on the line of code right under the imports in quotations where it says "r'/content/Loan_Data.csv'
"df_raw = pd.read_csv(r'/content/Loan_Data.csv') #614x13 size"

After that you too can run the model get it trained. Mess around with the parameters and see if you can get it to a higher accuracy. Else you can train it and test it on dummy or real sample data and see if it correctly predicts loans you or people you know might have applied for.
