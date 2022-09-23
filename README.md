# Recommender System based on Amazon reviews
This project will be about using machine learning to recommend products to users that made amazon reviews. Based off of the sentiment of review, different products will be recommended to the user.

## This repo contains:

**Saved_Models Folder** - Models used in this repo are saved in this folder and can be loaded into the respective notebooks denoted by the number at the beginning of the file names.

**1a_Preprocessing_and_EDA** - Contains preprocessing and EDA for the amazon review data.  
**1b_Meta_Data_Cleaning** - Contains preprocessing and EDA for the product metadata.  
**2_Modelling_Without_Text_Data** - Contains three ML models to determine if numeric features excluding the text can be used to predict the review sentiment.  
**3_NLP_Analysis** - Contains three ML models to use the review text to predict the review sentiment.  
**4_Recommendation_System** - Contains 1 recommendation system to recommend products based off of review sentiment and the product description.  

**environment.yml** - Saved environment dependencies used for this repo

## Workflow to replicate results:

**1. Setup environment**

    conda env create -f environment.yml


**2. Obtain data**

**3. Run workbooks in order**  
    From preprocessing and EDA first to Recommendation systems last  
    1a -> 1b -> 2 -> 3 -> 4  


Future works
1. Data pipeline
2. Website for user to put in sample reviews for a product and receive a recommendation



