import logging
import src.data.preprocessing as dpp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

target_feature = 'price'
    
feature_selection_listings = ['id', 'amenities', 
    'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 
    'host_is_superhost', 'host_listings_count', 'host_has_profile_pic', 'host_identity_verified', 
    'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude', 'longitude', 
    'property_type', 'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 
    'price', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_365', 
    'number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy', 
    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 
    'review_scores_location', 'review_scores_value', 'instant_bookable', 'reviews_per_month',
]

review_scores_features = [
    'review_scores_accuracy', 'review_scores_checkin', 'review_scores_cleanliness',
    'review_scores_rating', 'review_scores_value', 'review_scores_location', 
    'reviews_per_month', 'review_scores_communication'
]

sentiment_features = [
    'median_sentiment', 'min_sentiment', 'max_sentiment',
    'std_sentiment', 'sentiment_range', 'pct_negative', 'total_reviews_scored',
    'days_since_last_review', 'review_velocity', 'recent_mean_sentiment',
    'sentiment_trend', 'mean_sentiment'
]

'''
- Load a train and test set
- Train set contains only samples with and without review_scores
- Add a flag indicating whether a sample has reviews scores or not
- Fill missing review_scores with 0
'''
def load_A1(listings_df, sentiments_df, date_of_scrape, test_size=0.2, random_state=42):
    logger.info("Loading A1")

    # Merge listings and review data
    df = pd.merge(listings_df, sentiments_df, left_on='id', right_on='listing_id', how='outer')

    # Select features
    df = dpp.select_features(df, feature_selection_listings + sentiment_features)

    # Convert price from string to numerical value and drop samples with missing prices and remove outliers
    df = dpp.price(df, 2000)

    # Add flag for review_scores and fill missing values for review scores with 0.
    df = dpp.create_has_all_features_flag(df, review_scores_features + sentiment_features, 'review_scores')

    # Split into train and test sets
    X_train, X_test, y_train, y_test = dpp.split(df, target_feature, test_size, random_state)

    # Preprocess features
    X_train, X_test, y_train, y_test = preprocess_features(X_train, X_test, y_train, y_test, date_of_scrape)

    
    return X_train, X_test, y_train, y_test


'''
- Load train and test set
- Train set contains only samples with review_scores
'''
def load_A2(listings_df, sentiments_df, date_of_scrape, test_size=0.2, random_state=42):
    logger.info("Loading A2...")
    
    # Merge listings and review data
    df = pd.merge(listings_df, sentiments_df, left_on='id', right_on='listing_id', how='outer')

    # Select features
    df = dpp.select_features(df, feature_selection_listings + sentiment_features)

    # Convert price from string to numerical value and drop samples with missing prices and remove outliers
    df = dpp.price(df, 2000)


    # Split into train and test sets
    X_train, X_test, y_train, y_test = dpp.split(df, target_feature, test_size, random_state)

    # Drop samples without review_scores from the train_set
    X_train, y_train = dpp.drop_samples_missing_features(X_train, y_train, review_scores_features + sentiment_features)

    # Preprocess all features
    X_train, X_test, y_train, y_test = preprocess_features(X_train, X_test, y_train, y_test, date_of_scrape)
    
    # Fill missing review_scores features
    X_train, X_test = dpp.median_fill(X_train, X_test, review_scores_features + sentiment_features)

    return X_train, X_test, y_train, y_test

'''
- Load train and test set
- Train set contains only samples without review_scores
- Drop all review_scores columns
'''
def load_B1(listings_df, date_of_scrape, test_size=0.2, random_state=42):
    logger.info("Loading B1...")
    df = dpp.select_features(listings_df, feature_selection_listings)

    # Convert price from string to numerical value and drop samples with missing prices
    df = dpp.price(df, 2000)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = dpp.split(df, target_feature, test_size, random_state)

    # Drop samples with reviews
    X_train, y_train = dpp.drop_samples_with_features(X_train, y_train, review_scores_features)

    # Preprocess all features
    X_train, X_test, y_train, y_test = preprocess_features(X_train, X_test, y_train, y_test, date_of_scrape)

    # Drop review_scores columns
    X_train = X_train.drop(columns=review_scores_features)
    X_test = X_test.drop(columns=review_scores_features)

    return X_train, X_test, y_train, y_test

'''
- Load train and test set
- Train set contains with and without review_scores
- Drop all review_scores columns
'''
def load_B2(listings_df, date_of_scrape, test_size=0.2, random_state=42):
    logger.info("Loading B2...")
    df = dpp.select_features(listings_df, feature_selection_listings)

    # Convert price from string to numerical value and drop samples with missing prices
    df = dpp.price(df, 2000)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = dpp.split(df, target_feature, test_size, random_state)

    # Preprocess all features
    X_train, X_test, y_train, y_test = preprocess_features(X_train, X_test, y_train, y_test, date_of_scrape)

    # Drop review_scores columns
    X_train = X_train.drop(columns=review_scores_features)
    X_test = X_test.drop(columns=review_scores_features)
    
    return X_train, X_test, y_train, y_test


'''
- Load train and test set
- Train set contains samples with and without review_scores
- Drop all review_scores columns
'''
def preprocess_features(X_train, X_test, y_train, y_test, date_of_scrape):
    logger.info("Preprocessing features...")
    # host_since
    X_train, X_test = dpp.host_since(X_train, X_test, date_of_scrape)

    preprocessing_pipeline = [
        dpp.host_response_time_ordinal_encoding,
        dpp.host_response_rate, dpp.host_acceptance_rate, dpp.host_is_superhost,
        dpp.host_listings_count, dpp.host_has_profile_pic, dpp.host_identity_verified,
        dpp.instant_bookable, dpp.beds_bedrooms, dpp.reviews_per_month,
        dpp.room_type, dpp.bathrooms_text, dpp.neighbourhood_group_cleansed,
        dpp.fill_remaining_features, dpp.availability_ratio,
        dpp.beds_per_accommodates, dpp.distance_to_popular_landmarks
    ]

    for preproceesing_func in preprocessing_pipeline:
        logger.info(f"Shape: X_train={X_train.shape}, X_test={X_test.shape}")
        X_train, X_test = preproceesing_func(X_train, X_test)
    logger.info(f"Shape: X_train={X_train.shape}, X_test={X_test.shape}")

    # One Hot Encode 'amenities' using Multi Label Binarizer
    X_train, X_test = dpp.amenities(X_train, X_test, y_train)

    # Target encoding features
    X_train, X_test = dpp.property_type(X_train, X_test, y_train)
    X_train, X_test = dpp.neighbourhood_cleansed(X_train, X_test, y_train)

    # Log-Transform the target column
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    # Drop the id column
    X_train, X_test = dpp.drop_id(X_train, X_test)

    return X_train, X_test, y_train, y_test
