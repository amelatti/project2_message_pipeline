# Project_2_Message_Pipeline
 Udacity Data Science Course - Project 2 Emergency message data pipeline
 A project with the goal of training a Machine Learning model (via a data pipeline) on emergency message data to help identify future requests in a web app format.
### Table of Contents

1. [Installation](#installation)
2. [File Descriptions](#files)
3. [How to Use](#use)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Necessary libraries to run the code include: pandas, numpy, sklearn, plotly, flask, sqalchemy, ntlk, and json.  The code has been run on Python 3.8.

## File Descriptions <a name="files"></a>

1) .py file process_data.py includes the code that was used to import .csvs of the Disaster Message Data, combine them, and store them in a SQL .db
2) .py file train_classifier.py includes the code that was used to import data from the SQL .db, clean it and tokenize it, train and test a model with it, and then export the  model.
3) .py file run.py includes the code that is used to host the flask web app which holds data visualizations as well as an interactive form of the ML model


## How to Use <a name="use"></a>

In a command terminal run the following:
1) Run the Extract, transform, load pipeline to import .csvs of the Disaster Message Data, combine them, and store them in a SQL .db
    'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
2) Run the Machine Learning pipeline to train a model off of the SQL .db data, test the accuracy, and then export the model to a .pkl file
     'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
3) Navigate to the app folder and run the web app by entering 'python run.py'
    Paste the generated web address in your browser to view the app.

## Results<a name="results"></a>

The key results of the project can be found in the application, however there are a few things to note:
1) The GridSearchCV portion of the pipeline takes a long time to train the model, depending on how many parameters you choose to include for modification!
    NOTE: for my final code, I opted to leave a single parameter in so that runtime would not be hindered. 
2) The model has more accuracy for some classification categories than others. This could be related to key token identification.
    Specifically, if somebody needs water there may be more information about water in the message.
    But if somebody doesn't mention water in their message, then the model must learn that thirst, parched, overheating may all be key tokens for identifying the water category.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The Udacity Data Science Course provided the code templates and examples for many of the items in this project, especially the flask web app development. Please consider checking out the course if you found this project interesting. [here](https://www.udacity.com/courses/all?utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=747168232_c_individuals&utm_term=126315200811&utm_keyword=udacity_e&gclid=CjwKCAjw586hBhBrEiwAQYEnHVqnxwSRVfaDb53lwF5Fa3Jx1xeR7nfh3ZokP82uTh0IFPzLBeHE0RoC5RsQAvD_BwE).
As for my personal code, you can use it to your heart's content!