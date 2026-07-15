import logging
import time
import sys
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('..'))
from src.data.loading import load_A1, load_A2, load_B1, load_B2

logger = logging.getLogger(__name__)

def set_up():
    listings_df = pd.read_csv('../data/listings.csv.gz', low_memory=False)
    reviews_df = pd.read_csv('../data/reviews.csv.gz', low_memory=False)

    # TODO Add preprocessing of reviews_df here to create sentiment_df

    sentiments_df = pd.read_csv('../data/sentiment_features_unweighted.csv', low_memory=False)

    date_of_scrape = pd.to_datetime(listings_df["last_scraped"][0])

    X_train, X_test, y_train, y_test = load_A1(listings_df, sentiments_df, date_of_scrape)
    
    logger.debug(f'Shape A1_X_train: {X_train.shape}')
    logger.debug(f'Shape A1_X_test: {X_test.shape}')
    logger.debug(f'Shape A1_y_train: {y_train.shape}')
    logger.debug(f'Shape A1_y_test: {y_test.shape}')

    X_train.to_parquet(path='../data/processed/A1/X_train.parquet', engine='pyarrow')
    X_test.to_parquet(path='../data/processed/A1/X_test.parquet', engine='pyarrow')
    y_train.to_frame().to_parquet(path='../data/processed/A1/y_train.parquet', engine='pyarrow')
    y_test.to_frame().to_parquet(path='../data/processed/A1/y_test.parquet', engine='pyarrow')

    X_train, X_test, y_train, y_test = load_A2(listings_df, sentiments_df, date_of_scrape)
    
    logger.info(f'Shape A2_X_train: {X_train.shape}')
    logger.info(f'Shape A2_X_test: {X_test.shape}')
    logger.info(f'Shape A2_y_train: {y_train.shape}')
    logger.info(f'Shape A2_y_test: {y_test.shape}')

    X_train.to_parquet(path='../data/processed/A2/X_train.parquet', engine='pyarrow')
    X_test.to_parquet(path='../data/processed/A2/X_test.parquet', engine='pyarrow')
    y_train.to_frame().to_parquet(path='../data/processed/A2/y_train.parquet', engine='pyarrow')
    y_test.to_frame().to_parquet(path='../data/processed/A2/y_test.parquet', engine='pyarrow')

    X_train, X_test, y_train, y_test = load_B1(listings_df, date_of_scrape)
    
    logger.info(f'Shape B1_X_train: {X_train.shape}')
    logger.info(f'Shape B1_X_test: {X_test.shape}')
    logger.info(f'Shape B1_y_train: {y_train.shape}')
    logger.info(f'Shape B1_y_test: {y_test.shape}')
    
    X_train.to_parquet(path='../data/processed/B1/X_train.parquet', engine='pyarrow')
    X_test.to_parquet(path='../data/processed/B1/X_test.parquet', engine='pyarrow')
    y_train.to_frame().to_parquet(path='../data/processed/B1/y_train.parquet', engine='pyarrow')
    y_test.to_frame().to_parquet(path='../data/processed/B1/y_test.parquet', engine='pyarrow')

    X_train, X_test, y_train, y_test = load_B2(listings_df, date_of_scrape)
    
    logger.info(f'Shape B2_X_train: {X_train.shape}')
    logger.info(f'Shape B2_X_test: {X_test.shape}')
    logger.info(f'Shape B2_y_train: {y_train.shape}')
    logger.info(f'Shape B2_y_test: {y_test.shape}')
    
    X_train.to_parquet(path='../data/processed/B2/X_train.parquet', engine='pyarrow')
    X_test.to_parquet(path='../data/processed/B2/X_test.parquet', engine='pyarrow')
    y_train.to_frame().to_parquet(path='../data/processed/B2/y_train.parquet', engine='pyarrow')
    y_test.to_frame().to_parquet(path='../data/processed/B2/y_test.parquet', engine='pyarrow')


if __name__ == "__main__":
    logger.info('Setting up datasets...')
    set_up()