import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

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