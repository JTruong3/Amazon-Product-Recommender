import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

def class_report(model,X_test_ss,y_test):
    '''
    This function takes in the machine learning model as well as the test data and outputs a classification report.
    
    '''
    
    # Get the model predictions.
    y_pred = model.predict(X_test_ss)

    # Get the classification report for the model
    c_report = classification_report(y_test,y_pred)
    
    return print(c_report)


def conf_matrix(model,X_test_ss,y_test,model_type):
    '''
    This function takes in the machine learning model, test data and the model name. 
    The output is a decision matrix for the model
    
    '''
    
    # Plot the decision matrix
    plot_confusion_matrix(model, X_test_ss, y_test, cmap = 'Blues', values_format ='')

    # Add labels to the plot
    plt.title(f'Decision Matrix for the {model_type} model')
    plt.show()


def rec_system(vectorizer, sentiment, product_list, product_name):
    '''
    
    The purpose of this function is to recommend products based on the predicted
    sentiment and product name.
    
    The inputs to this function include the vectorizer model, the sentiment of 
    the review text, the list of products, product name to be used for 
    recommending other products.
    
    
    Parameters
    ----------
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer,
                 Vectorizer model that was trained on the product descriptions
    sentiment = int, 
                A positive or negative sentiment prediction
    product_list = pandas.core.frame.DataFrame,
                   A Dataframe of the products with the descriptions
    product_name = str,
                   The name of the movie for comparison.
    
    
    Returns
    --------
    Recommended Product list
    
    '''
    
    # Transform the description
    TF_matrix = vectorizer.transform(product_list['description_0'])
    
    # Determine the index of the product of recommendation
    product_index = product_list[product_list['title'] == product_name].index
    
    # Get the TF matrix of the required product
    TF_matrix_product = TF_matrix[product_index]
    
    # Determine the similarity between the product and everything else
    product_similarities = cosine_similarity(TF_matrix,TF_matrix_product, dense_output = False)
    
    # 
    single_df = pd.DataFrame({'item':product_list['title'], 
                       'similarities': np.array(product_similarities.todense()).squeeze()})
    
    # Sort the products by similarity
    single_df_sorted = single_df.sort_values(by = 'similarities', ascending = False)
    
    
    # When the sentiment is positive, recommend the top 10 most similar products
    if sentiment == 1:
        recommended_products = single_df_sorted.head(10)
    
    # Recommend 10 products randomly from the top 100 most similar products
    else:
        # Split out the top 100 recommended_products
        top_100 = single_df_sorted.iloc[:100].reset_index(drop = True)
        
        # Randomly sample 10 numbers between 0 and 99
        ran_num = random.sample(range(0, 100), 10)
        
        # Find the index where the ran_num exists in the top 100 recommended products
        top_100_index = top_100.index.isin(ran_num)
        
        # Determine recommended products
        recommended_products = top_100[top_100_index]
        
    return recommended_products.reset_index(drop = True)