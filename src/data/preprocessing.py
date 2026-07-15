import ast
import re
import logging
import pandas as pd
import numpy as np
import category_encoders as ce
from haversine import haversine
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# TODO: Functions for filling of latitude and longitude | Distance to famous landmarks

''' 
- Select features from a DataFrame
- Returns DataFrame with selected features
'''
def select_features(df, features):
    logger.debug(f"Select {len(features)} features...")
    df_selected = df[features].copy()
    logger.debug(f"DataFrame Shape after feature selection: {df_selected.shape}")
    return df_selected

'''
- Add a flag as a new feature indicating whether a sample has all review_scores features.
- Fill missing review_scores with 0
'''
def create_has_all_features_flag(df, features, name):
    logger.debug(f"Create 'has_all_{name}' flag...")
    df[f'has_all_{name}'] = (df[features].isna().sum(axis=1) == 0).astype(int)
    
    perfect_scores = df[f'has_all_{name}'].sum()
    logger.debug(f"{perfect_scores} of {len(df)} have all review scores.")
    
    logger.debug("Fill NA review_scores with 0.0 ...")
    df[features] = df[features].fillna(0.)
    return df

'''
- Drop all rows that have a NA for any feature given.
'''
def drop_samples_missing_features(X, y, features):
    initial_len = len(X)
    logger.info(f"Filter: remove samples, that miss any of the following features: {features}...")
    
    drop_condition = X[features].isna().any(axis=1)
    X = X[~drop_condition]
    y = y[~drop_condition]
    
    dropped_count = initial_len - len(X)
    logger.info(f"{dropped_count} Samples dropped. Updated shape: X={X.shape}, y={y.shape}")
    
    return X, y

'''
- Drop all rows, that have no missing values for a selections of features
'''
def drop_samples_with_features(X, y, features):
    initial_len = len(X)
    logger.info(f"Filter: Remove samples, that have no missing values for the following features: {features}...")
    
    drop_condition = X[features].isna().sum(axis=1) == 0 
    X = X[~drop_condition]
    y = y[~drop_condition]
    
    dropped_count = initial_len - len(X)
    logger.info(f"{dropped_count} samples dropped. Updated shape: X={X.shape}, y={y.shape}")
    
    return X, y

def price(df, outlier=None):
    target_feature = 'price'
    
    logger.info(f"Processing target feature '{target_feature}': {df.shape}")
    initial_len = len(df)
    
    # Drop NA
    df = df.dropna(subset=[target_feature]).copy()

    dropped = initial_len - len(df)
    logger.info(f"Dropped {dropped} samples missing the target feature '{target_feature}'.")

    # Convert string to numerical value
    logger.debug(f"Converting target feature '{target_feature}' to numerical values...")
    df[target_feature] = df[target_feature].str.replace(r"[$,]", "", regex=True).astype("float")

    if outlier is not None:
        drop_cond = df[target_feature] >= outlier
        logger.debug(f'Drop {drop_cond.sum()} outliers with price >= {outlier}')
        df = df[~drop_cond]

    return df

'''
- Split DataFrame into a train and test set
'''
def split(df, target_feature, test_size=0.2, random_state=42):
    logger.info(f"Preparing to split dataset. Initial shape: {df.shape}")
    
    # Split features from target
    X = df.drop(columns=target_feature)
    y = df[target_feature]

    # Split into train and test sets
    logger.info(f"Performing train-test split (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logger.info(f"Split complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

'''
- Drop samples that have a certain amount of NA
- missing_values: Amount of NA a sample needs to have to be dropped
'''
def drop_vague_samples(X, y, missing_values):
    initial_len = len(X)
    logger.info(f"Filtering: Dropping samples with {missing_values} or more missing values...")
    
    # Create a mask whether a sample has more than missing_values missing values
    drop_condition = X.isna().sum(axis=1) >= missing_values
    X = X[~drop_condition]
    y = y[~drop_condition]

    dropped_count = initial_len - len(X)
    logger.info(f"Dropped {dropped_count} vague samples. New shape: X={X.shape}, y={y.shape}")
    return X, y

#########################################
# Some utility functions for filling NA # 
#########################################

def median_fill(X_train, X_test, columns):
    logger.debug(f"Filling missing values with median for {columns}...")
    
    medians_train = X_train[columns].median()

    logger.debug(f"Medians: {dict(zip(columns, medians_train))}...")

    X_train[columns] = X_train[columns].fillna(medians_train)
    X_test[columns] = X_test[columns].fillna(medians_train)

    return X_train, X_test

def mode_fill(X_train, X_test, columns):
    logger.debug(f"Filling missing values with median for {columns}...")
    
    modes_train = X_train[columns].mode().iloc[0]
    
    logger.debug(f"Modes: {dict(zip(columns, modes_train))}...")

    X_train[columns] = X_train[columns].fillna(modes_train)
    X_test[columns]  = X_test[columns].fillna(modes_train)
    
    return X_train, X_test

def log_transform(df, columns):
    logger.debug(f"Applying log1p transformation to {columns}...")
    
    df[columns] = np.log1p(df[columns])
    return df

###################################################################################################################
# Now follows a collection of functions that convert the datatype of specific features or encode them numerically #
# The function names contain the name of the specific feature                                                     #
# They also fill missing values in the train and test sets                                                        #
###################################################################################################################

'''
Name: host_since
Description: Date of when a host became a host
Type: String
Initial format: 'yyyy.mm.dd'

Encoding: Number of days a host has been a host
Filling: Fill NA with the median calculated on the train set
'''
def host_since(X_train, X_test, date):
    logger.info("Processing feature: 'host_since'")
    
    # Convert date to the number of days
    logger.debug(f"Converting 'host_since' dates to number of days relative to {date}...")
    X_train['host_since'] = (date - pd.to_datetime(X_train["host_since"], errors="coerce")).dt.days
    X_test['host_since'] = (date - pd.to_datetime(X_test["host_since"], errors="coerce")).dt.days

    # Fill missing values with the median
    host_since_median_train = X_train['host_since'].median()
    logger.debug(f"Filling missing values in 'host_since' with train median: {host_since_median_train} days")

    X_train['host_since'] = X_train['host_since'].fillna(host_since_median_train)
    X_test['host_since'] = X_test['host_since'].fillna(host_since_median_train)
    return X_train, X_test

'''
Name: host_response_time
Description: The amount of time a host takes to response to requests. There are four classes of response times.
Type: String
Classes: "within an hour", "within a few hours", "within a day", "a few days or more"

Encoding: Ordinal Encoding
Filling: Fill NA with the mode calculated on the train set
'''
def host_response_time_ordinal_encoding(X_train, X_test, ranking=None):
    logger.info("Processing feature: 'host_response_time' (Ordinal Encoding)")

    # Fill NA with its mode
    X_train, X_test = mode_fill(X_train, X_test, ['host_response_time'])

    if ranking is None:
        ranking = {
            "within an hour": 1,
            "within a few hours": 5,
            "within a day": 12,
            "a few days or more": 36
        }

    logger.debug(f"Using ranking mapping for 'host_response_time': {ranking}")

    mapping = {
        'col': 'host_response_time',
        'mapping': ranking
    }

    # Initialise the encoder
    logger.debug("Fitting OrdinalEncoder on the train set...")
    oe = ce.ordinal.OrdinalEncoder(
        cols=['host_response_time'], 
        mapping=[mapping], 
        return_df=True, 
        handle_missing='return_nan',
        handle_unknown='return_nan'
    )

    # Fit the encoder on the train set
    oe.fit(X_train)

    # Transform the columns
    logger.debug("Applying ordinal transformation to train and test sets...")
    X_train = oe.transform(X_train)
    X_test = oe.transform(X_test)


    return X_train, X_test

'''
Name: host_response_rate
Description: The relative amount of times a host responds to requests
Type: String
Format: 'x %', where x is a positive number between 0 and 100

Encoding: Extract x and divide by 100
Filling: Fill NA with the median calculated on the train set
'''
def host_response_rate(X_train, X_test):
    logger.info("Processing feature: 'host_response_rate'")

    # Convert rate to float
    logger.debug("Removing '%' and converting 'host_response_rate' to float (ratio 0-1)...")
    X_train['host_response_rate'] = X_train['host_response_rate'].str.replace("%", "").astype("float") / 100
    X_test['host_response_rate'] = X_test['host_response_rate'].str.replace("%", "").astype("float") / 100

    # Fill NA
    X_train, X_test = median_fill(X_train, X_test, ['host_response_rate'])
    
    return X_train, X_test

'''
Name: host_response_rate
Description: The relative amount a hosts accepts a request.
Type: String
Format: 'x %', where x is a positive number between 0 and 100

Encoding: Extract x and divide by 100
Filling: Fill NA with the median calculated on the train set
'''
def host_acceptance_rate(X_train, X_test):
    logger.info("Processing feature: 'host_acceptance_rate'")
    
    # Convert rate to float
    logger.debug("Removing '%' and converting 'host_acceptance_rate' to float (ratio 0-1)...")
    X_train['host_acceptance_rate'] = X_train['host_acceptance_rate'].str.replace("%", "").astype("float") / 100
    X_test['host_acceptance_rate'] = X_test['host_acceptance_rate'].str.replace("%", "").astype("float") / 100

    # Fill NA
    X_train, X_test = median_fill(X_train, X_test, ['host_acceptance_rate'])
    
    return X_train, X_test

'''
Name: host_is_superhost
Description: Indicates whether the host is a superhost or not.
Type: String
Format: 't' (host is a superhost), 'f' (host is not a superhost)

Encoding: {'t': 1, 'f': 0}
Filling: Fill NA with the mode calculated on the train set
'''
def host_is_superhost(X_train, X_test):
    logger.info("Processing feature: 'host_is_superhost'")
    mapping = {
    "t": 1,
    "f": 0
    }

    logger.debug(f"Mapping 'host_is_superhost' string values to integers using {mapping}...")
    X_train['host_is_superhost'] = X_train['host_is_superhost'].map(mapping)
    X_test['host_is_superhost'] = X_test['host_is_superhost'].map(mapping)
    
    X_train, X_test = mode_fill(X_train, X_test, columns=['host_is_superhost'])

    return X_train, X_test


'''
Name: host_listings_count
Description: Indicates how many a host created.
Type: Integer

Filling: Fill NA with the median calculated on the train set
'''
def host_listings_count(X_train, X_test):
    logger.info("Processing feature: 'host_listings_count'")
    X_train, X_test = median_fill(X_train, X_test, ['host_listings_count'])
    return X_train, X_test

'''
Name: host_has_profile_pic
Description: Indicates whether the host has a profile uploaded or not.
Type: String
Format: 't' (host has a profile picture uploaded), 'f' (host hasn't uploaded a profile picture)

Encoding: {'t': 1, 'f': 0}
Filling: Fill NA with the mode calculated on the train set
'''
def host_has_profile_pic(X_train, X_test):
    logger.info("Processing feature: 'host_has_profile_pic'")
    mapping = {
    "t": 1,
    "f": 0
    }

    logger.debug(f"Mapping 'host_has_profile_pic' string values to integers using {mapping}...")
    X_train['host_has_profile_pic'] = X_train['host_has_profile_pic'].map(mapping)
    X_test['host_has_profile_pic'] = X_test['host_has_profile_pic'].map(mapping)
    
    X_train, X_test = mode_fill(X_train, X_test, columns=['host_has_profile_pic'])

    return X_train, X_test

'''
Name: host_identity_verified
Description: Indicates whether the hosts identity is verified or not.
Type: String
Format: 't' (host is verified), 'f' (host is not verified)

Encoding: {'t': 1, 'f': 0}
Filling: Fill NA with the mode calculated on the train set
'''
def host_identity_verified(X_train, X_test):
    logger.info("Processing feature: 'host_identity_verified'")
    mapping = {
    "t": 1,
    "f": 0
    }

    logger.debug(f"Mapping 'host_identity_verified' string values to integers using {mapping}...")
    X_train['host_identity_verified'] = X_train['host_identity_verified'].map(mapping)
    X_test['host_identity_verified'] = X_test['host_identity_verified'].map(mapping)
    
    X_train, X_test = mode_fill(X_train, X_test, columns=['host_identity_verified'])

    return X_train, X_test

'''
Name: instant_bookable
Description: Indicates whether the hosts identity is verified or not.
Type: String
Format: 't' (instant bookable), 'f' (not instant bookable)

Encoding: {'t': 1, 'f': 0}
Filling: Fill NA with the mode calculated on the train set
'''
def instant_bookable(X_train, X_test):
    logger.info("Processing feature: 'instant_bookable'")
    mapping = {
    "t": 1,
    "f": 0
    }

    logger.debug(f"Mapping 'instant_bookable' string values to integers using {mapping}...")
    X_train['instant_bookable'] = X_train['instant_bookable'].map(mapping)
    X_test['instant_bookable'] = X_test['instant_bookable'].map(mapping)
    
    X_train, X_test = mode_fill(X_train, X_test, columns=['instant_bookable'])

    return X_train, X_test

'''
Names: beds & bedrooms
Description: Indicate how many beds and how many bedrooms a listings got.
Type: Integer

Filling: Fill NA with the median calculated on the train set
'''
def beds_bedrooms(X_train, X_test):
    logger.info("Processing features: 'beds' and 'bedrooms'")
    return median_fill(X_train, X_test, ['beds', 'bedrooms'])

'''
Names: review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, review_scores_rating
Description: The average user rating a listing received for certain aspects.
Type: Integer

Filling: Fill NA with the median calculated on the train set
'''
def review_scores(X_train, X_test):
    logger.info("Processing features: 'review_scores_*'")
    columns=[
        'review_scores_accuracy', 'review_scores_cleanliness', 
        'review_scores_checkin', 'review_scores_communication', 
        'review_scores_location', 'review_scores_value', 'review_scores_rating'
    ]
    return median_fill(X_train, X_test, columns)

'''
Name: reviews_per_month
Description: The average number of reviews per month over the lifetime of the listing.
Type: Integer

Filling: Fill NA with the median calculated on the train set
'''
def reviews_per_month(X_train, X_test):
    logger.info("Processing feature: 'reviews_per_month'")
    return median_fill(X_train, X_test, ['reviews_per_month'])

'''
Name: room_type
Description: What kind of room type the listing is. There are four classes of room types.
Type: String
Classes: 'Entire home/apt' 'Private room' 'Shared room' 'Hotel room'

Encoding: One hot encoding
Filling: Fill NA with the mode calculated on the train set
'''
def room_type(X_train, X_test):
    logger.info("Processing feature: 'room_type' (One-Hot Encoding)")
    
    # Fill NA
    X_train, X_test = mode_fill(X_train, X_test, ['room_type']) # TODO: Maybe fill with a new category 'unknown room_type'

    # Initialise the OneHotEncoder
    logger.debug("Fitting OneHotEncoder for 'room_type'...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit and transform the encoder
    encoded_data_train = encoder.fit_transform(X_train[['room_type']])
    encoded_data_test = encoder.transform(X_test[['room_type']])
    
    # Clean room_type names
    raw_names = encoder.get_feature_names_out(['room_type'])
    clean_names = [re.sub(r'[ /]', '_', name).lower() for name in raw_names]
    logger.debug(f"Generated clean One-Hot features: {clean_names}")

    # Create DataFrames caontaining encoded data
    encoded_df_train = pd.DataFrame(
        data=encoded_data_train,
        columns=clean_names,
        index=X_train.index
    )

    encoded_df_test = pd.DataFrame(
        data=encoded_data_test,
        columns=clean_names,
        index=X_test.index
    )

    # Concatenate one hot encoded DataFrame with original DataFrames
    X_train = pd.concat([X_train.drop(columns='room_type'), encoded_df_train], axis=1)
    X_test = pd.concat([X_test.drop(columns='room_type'), encoded_df_test], axis=1)

    return X_train, X_test

'''
Name: bathrooms_text
Description: Contains a brief description about the type of the bathroom and in most cases the number of bathrooms. 
             Descriptions that don't contain a number still contain information about the number of bathrooms.
             For example there are 'half bathrooms'. (We convert these to 0.5).
             It also contains information whether the bathrooms are shared or private.
Type: String

Encoding: Extract the number (if present) and encode whether the bathroom is shared or private
Filling: Fill NA in amount of baths with median and shared baths with the mode.
'''
def bathrooms_text(X_train, X_test):
    logger.info("Processing feature: 'bathrooms_text' (Extracting numerical and categorical info)")
    bathroom_descriptions_train = X_train['bathrooms_text'].str.lower()
    bathroom_descriptions_test = X_test['bathrooms_text'].str.lower()

    # Encode whether a bath is shared or not
    logger.debug("Extracting 'shared_bath' information...")
    X_train['shared_bath'] = X_train['bathrooms_text'].str.lower().str.contains('shared', na=np.nan).astype('float')
    X_test['shared_bath'] = X_test['bathrooms_text'].str.lower().str.contains('shared', na=np.nan).astype('float')

    # Extract how many baths a listing has
    logger.debug("Extracting number of 'baths' using RegEx...")
    X_train['baths'] = bathroom_descriptions_train.str.extract(r'(\d+\.?\d*)').astype('float')
    X_test['baths'] = bathroom_descriptions_test.str.extract(r'(\d+\.?\d*)').astype('float')

    # Create masks to identify 'half baths'
    logger.debug("Applying special rule: Converting 'half baths' to 0.5...")
    halfbath_mask_train = bathroom_descriptions_train.str.contains('half', na=False)
    halfbath_mask_test = bathroom_descriptions_test.str.contains('half', na=False)
    
    # Encode 'half baths'
    X_train.loc[halfbath_mask_train, 'baths'] = 0.5
    X_test.loc[halfbath_mask_test, 'baths'] = 0.5

    # Drop original feature
    X_train = X_train.drop(columns='bathrooms_text')
    X_test = X_test.drop(columns='bathrooms_text')

    # Fill NA
    X_train, X_test = median_fill(X_train, X_test, ['baths'])
    X_train, X_test = mode_fill(X_train, X_test, ['shared_bath'])

    return X_train, X_test

'''
Name: property_type
Description: What kind of property the listing is. There are many categories of property types.
Type: String

Encoding: Target encoding (grading based on the mean price for each property type)
Filling: Fill NA with the mode calculated on the train set
'''
def property_type(X_train, X_test, y_train):
    logger.info("Processing feature: 'property_type' (Target Encoding)")
    
    # Fill NA
    X_train, X_test = mode_fill(X_train, X_test, ['property_type'])
    
    # Initialise encoder
    logger.debug("Fitting TargetEncoder (min_samples_leaf=10, smoothing=10)...")
    encoder = ce.TargetEncoder(cols=['property_type'], min_samples_leaf=10, smoothing=10, return_df=True)
    encoder.fit(X_train, y_train)

    ordinal_mapping = encoder.ordinal_encoder.category_mapping[0]['mapping']
    target_mapping = encoder.mapping['property_type']
    target_mapping = ordinal_mapping.map(target_mapping)
    logger.debug(f'Target encoding of property type: \n{target_mapping}')

    # Encode property types
    logger.debug("Applying target encoding transformation...")
    X_train = encoder.transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test

'''
Name: neighbourhood_group_cleansed
Description: The neighbourhood group a listing is part of. There are around 10 groups.
Type: String

Encoding: One Hot Encoding
Filling: Fill NA with the mode calculated on the train set
'''
def neighbourhood_group_cleansed(X_train, X_test):
    logger.info("Processing feature: 'neighbourhood_group_cleansed' (One-Hot Encoding)")
    
    # Fill NA
    X_train, X_test = mode_fill(X_train, X_test, ['neighbourhood_group_cleansed'])

    # Init encoder
    logger.debug("Fitting OneHotEncoder for 'neighbourhood_group_cleansed'...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit encoder
    encoder.fit(X_train[['neighbourhood_group_cleansed']])

    # Transform encoder
    encoded_data_train = encoder.transform(X_train[['neighbourhood_group_cleansed']])
    encoded_data_test = encoder.transform(X_test[['neighbourhood_group_cleansed']])

    group_names_raw = encoder.get_feature_names_out(['neighbourhood_group_cleansed'])
    group_names_clean = [re.sub(r'[ -]', '_', name).lower() for name in group_names_raw]
    logger.debug(f"Generated clean One-Hot features: {group_names_clean}")

    # Create DataFrames
    encoded_df_train = pd.DataFrame(
        data=encoded_data_train,
        columns=group_names_clean,
        index=X_train.index
    )

    encoded_df_test = pd.DataFrame(
        data=encoded_data_test,
        columns=group_names_clean,
        index=X_test.index
    )

    # Drop the original feature
    X_train = X_train.drop(columns='neighbourhood_group_cleansed')
    X_test = X_test.drop(columns='neighbourhood_group_cleansed')

    # Concat encoded DataFrames
    X_train = pd.concat(
        [X_train, encoded_df_train],
        axis=1
    )
    X_test = pd.concat(
        [X_test, encoded_df_test],
        axis=1
    )

    return X_train, X_test

'''
Name: neighbourhood_cleansed
Description: The neighbourhood a listing is part of. There are many neighbourhoods.
Type: String

Encoding: Target encoding (grading based on the mean price for each neightbourhood)
Filling: Fill NA with the mode calculated on the train set
'''
def neighbourhood_cleansed(X_train, X_test, y_train):
    logger.info("Processing feature: 'neighbourhood_cleansed' (Target Encoding)")
    
    # Fill NA
    X_train, X_test = mode_fill(X_train, X_test, ['neighbourhood_cleansed'])

    # Init encoder
    logger.debug("Fitting TargetEncoder (min_samples_leaf=10, smoothing=10)...")
    encoder = ce.TargetEncoder(cols=['neighbourhood_cleansed'], min_samples_leaf=10, smoothing=10, return_df=True)  # TODO: find better parameters

    # Fit encoder on the train set
    encoder.fit(X_train, y_train)

    # Encode
    logger.debug("Applying target encoding transformation...")
    X_train = encoder.transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test

'''
Name: amenites
Description: Each listing contains a list of ameneties it offers. 
             There are around 2000 unique amenities, but many amenities describe the same thing.
             For example many listings offer televisions of slightly diffrent sizes.
Type: String (Amenities of a sample could look like this: '['tv', 'wifi', 'kitchen', ...]'. 
              Its a string of a list, that will be converted to a list of strings.)

Filtering: Apply a mapping that groups amenities that are basicly the same.
Encoding: One Hot Encoding using a Multi Label Binarizer and then extract the top 10 of amenities.
          A usual One Hot Encoder would encode the whole list as one amenity ['tv', 'wifi', 'kitchen'] 
            whereas the MLB encodes each element 'tv', 'wifi', 'kitchen'.
          TODO: Top 10 should be based on correlation instead of frequency
'''
# Initialise a list of rules to determine how amenities are grouped
amenity_mapping_rules = [
    # (contains, map_to)
    ('wifi', 'wifi'),
    ('ethernet', 'wifi'),
    ('refrigerator', 'refrigerator'),
    ('shampoo', 'shampoo'),
    ('showergel', 'shampoo'),
    ('coffee', 'coffee'),
    ('gameconsole', 'gameconsole'),
    ('soap', 'soap'),
    ('aircond', 'airconditioner'),
    ('ac-splitty', 'airconditioner'),
    ('acunit', 'airconditioner'),
    ('central', 'airconditioner'),
    ('conditioner', 'conditioner'),
    ('netflix', 'streaming_service'),
    ('amazonprimevideo', 'streaming_service'),
    ('disney+', 'streaming_service'),
    ('hbomax', 'streaming_service'),
    ('appletv', 'streaming_service'),
    ('chromecast', 'streaming_service'),
    ('stove', 'stove'),
    ('oven', 'oven'),
    ('soundsystem', 'soundsystem'),
    ('inchtv', 'tv'),
    ('hdtv', 'tv'),
    ('firetv', 'tv'),
    ('exerciseequipment', 'exerciseequipment'),
    ('grill', 'outdoor_amenities'),
    ('barbecue', 'outdoor_amenities'),
    ('housekeeping', 'housekeeping'),
    ('cleaninga', 'housekeeping'),
    ('build', 'housekeeping'),
    ('baby', 'baby_stuff'),
    ('highchair', 'baby_stuff'),
    ('crib', 'baby_stuff'),
    ('changingtable', 'baby_stuff'),
    ('children', 'children'),
    ('pool', 'pool'),
    ('hottub', 'pool'),
    ('clothingstorage', 'clothingstorage'),
    ('games', 'games'),
    ('paidwasher', 'paidwasher'),
    ('freewasher', 'freewascher'),
    ('washer', 'washer'),
    ('evcharger', 'evcharger'),
    ('fan', 'fan'),
    ('privateback', 'private_backyard'),
    ('sharedback', 'shared_backyard'),
    ('backyard', 'backyard'),
    

    (re.compile(r'paid.*parking'), 'paid_parking'),
    (re.compile(r'free.*parking'), 'free_parking'),
    (re.compile(r'private.*gym'), 'private_gym'),
    

    ('freeresidentialgarage', 'free_parking'),
    ('freecarport', 'free_parking'),
    ('gym', 'gym'),
    ('view', 'nice_view'),
    ('outdoor', 'outdoor_amenities'),
    ('hammock', 'outdoor_amenities'),

    ('indoorfireplace', 'indoorfireplace'),
    ('beach', 'beach'),
    ('tv', 'tv'),
    ('foosball', 'activities_nearby'),
    ('minigolf', 'activities_nearby'),
    ('climbing', 'activities_nearby'),
    ('hockey', 'activities_nearby'),
    ('skateramp', 'activities_nearby'),
    ('lasertag', 'activities_nearby'),
    ('bowling', 'activities_nearby'),
    ('lakeaccess', 'activities_nearby'),
    ('kayak', 'activities_nearby'),
    ('ski-in', 'activities_nearby'),
    ('sauna', 'activities_nearby'),
    ('sharedsau', 'activities_nearby'),
    ('pingpong', 'activities_nearby'),
    ('moviet', 'activities_nearby'),
    ('bikes', 'activities_nearby'),
    ('heat', 'heating'),
    
    # Exclude hair
    (('dryer', 'hair'), 'dryer')
]

# Define a function that applies the rules on a list
# The function takes a list of amenities and maps each element based on the rules
def group_amenities(amenities_list, amenity_mapping_rules):
    mapped_list = []
    
    for amenity in amenities_list:
        found = False
        
        for rule, target in amenity_mapping_rules:
            if isinstance(rule, str) and rule in amenity:
                mapped_list.append(target)
                found = True
                break
            elif isinstance(rule, re.Pattern) and rule.search(amenity):
                mapped_list.append(target)
                found = True
                break
            elif isinstance(rule, tuple):
                inc, exc = rule
                if inc in amenity and exc not in amenity:
                    mapped_list.append(target)
                    found = True
                    break
        
        if not found:
            mapped_list.append(amenity)
            
    return mapped_list

def amenities(X_train, X_test, y_train, rules=amenity_mapping_rules):
    logger.info("Processing feature: 'amenities' (Multi-Label Binarization & Mapping)")
    
    # Convert string of list to list of strings
    X_train['amenities'] = X_train['amenities'].str.lower().replace(r'[ ]', '', regex=True).apply(ast.literal_eval)
    X_test['amenities'] = X_test['amenities'].str.lower().replace(r'[ ]', '', regex=True).apply(ast.literal_eval)

    # Apply the mapping
    logger.debug("Grouping amenities using the defined mapping rules...")
    X_train['amenities'] = X_train['amenities'].apply(group_amenities, args=(rules, ))
    X_test['amenities'] = X_test['amenities'].apply(group_amenities, args=(rules, ))

    # Initialise multi label binarizer
    logger.debug("Fitting MultiLabelBinarizer...")
    encoder = MultiLabelBinarizer()

    # Fit and transform
    encoded_amenities_train = encoder.fit_transform(X_train['amenities'])
    encoded_amenities_test = encoder.transform(X_test['amenities'])
    
    # Convert to DataFrames
    encoded_amenities_df_train = pd.DataFrame(encoded_amenities_train, columns=encoder.classes_, index=X_train.index)
    encoded_amenities_df_test = pd.DataFrame(encoded_amenities_test, columns=encoder.classes_, index=X_test.index)
    
    # Concat new columns to the original data
    X_train = pd.concat([X_train, encoded_amenities_df_train], axis=1)
    X_test = pd.concat([X_test, encoded_amenities_df_test], axis=1)


    # Retrieve top 12 amenities based on correlation (pearson) with the target
    amenity_cols = [col for col in encoder.classes_ if col in X_train.columns]
    
    correlations = X_train[amenity_cols].corrwith(y_train)
    top_12_amenities = correlations.abs().sort_values(ascending=False).head(12).index.tolist()
    amenities_to_drop = [col for col in amenity_cols if col not in top_12_amenities]

    logger.info(f'Filtering amenities: Dropping {len(amenities_to_drop)} least correlated features to keep Top 12...')
    logger.info(f'Keeping amenities: {top_12_amenities}')
    X_train = X_train.drop(columns=amenities_to_drop)
    X_test = X_test.drop(columns=amenities_to_drop) 

    # Drop original feature
    X_train = X_train.drop(columns=['amenities'])
    X_test = X_test.drop(columns=['amenities'])

    return X_train, X_test


'''
Names: accommodates, minimum_nights, maximum_nights, availability_30, availability_365, number_of_reviews, number_of_reviews_ltm
Description: Remaining features.

Encoding: These features are already numerical.
Filling: Fill NA with the median calculated on the train set TODO: Does it make sense to fill these with their medians?
'''
def fill_remaining_features(X_train, X_test):
    remaining_features = [
        'accommodates', 'minimum_nights', 'maximum_nights', 
        'availability_30', 'availability_365', 'number_of_reviews',
        'number_of_reviews_ltm'
    ]
    logger.info(f"Processing remaining numerical features: {remaining_features}...")

    return median_fill(X_train, X_test, remaining_features)

def drop_id(X_train, X_test):
    logger.info("Dropping 'id' column from datasets...")
    # Drop id column
    X_train = X_train.drop(columns='id')
    X_test = X_test.drop(columns='id')
    return X_train, X_test


#######################
# Feature Engineering #
#######################

def availability_ratio(X_train, X_test):
    logger.info("Feature Engineering: Creating 'availability_ratio'...")
    X_train['availability_ratio'] = X_train['availability_365'] / 365
    X_test['availability_ratio'] = X_test['availability_365'] / 365
    return X_train, X_test

def beds_per_accommodates(X_train, X_test):
    logger.info("Feature Engineering: Creating 'beds_per_accommodates'...")
    X_train['beds_per_accommodates'] = X_train['beds'] / X_train['accommodates']
    X_test['beds_per_accommodates'] = X_test['beds'] / X_test['accommodates']
    return X_train, X_test

# Calculates haversine distances of each listing to popular landmarks
def distance_to_popular_landmarks(X_train, X_test):
    logger.info("Feature Engineering: Calculating haversine distances to popular landmarks in Barcelona...")
    landmarks = {
        'sagrada_familia': (41.4036, 2.1744),
        'barceloneta': (41.3784, 2.1925),
        'las_ramblas': (41.3810, 2.1730),
        'camp_nou': (41.3809, 2.1228),
        'placa_catalunya': (41.3870, 2.1700)
    }

    for key in landmarks.keys():
        logger.debug(f"Calculating distance to {key}...")
        X_train['distance_to_' + key] = X_train.apply(lambda listing: haversine(landmarks[key], (listing['latitude'], listing['longitude'])), axis=1)
        X_test['distance_to_' + key] = X_test.apply(lambda listing: haversine(landmarks[key], (listing['latitude'], listing['longitude'])), axis=1)

    return X_train, X_test
    