import pandas as pd

def add_high_amount_flag(df, threshold=220):
    """
    Adds a binary feature 'is_high_amount' based on TX_AMOUNT threshold.
    """
    df['is_high_amount'] = df['TX_AMOUNT'] > threshold
    return df


def add_terminal_fraud_count(df, window_days=28):
    """
    Adds a rolling fraud count per terminal over a time window.
    Much faster using groupby + rolling.
    """
    df = df.sort_values('TX_DATETIME')
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

    # Create a binary fraud flag
    df['is_fraud'] = df['TX_FRAUD'] == 1

    # Set datetime as index for rolling
    df.set_index('TX_DATETIME', inplace=True)

    # Group by terminal and apply rolling sum
    terminal_rolling = (
        df.groupby('TERMINAL_ID')['is_fraud']
        .rolling(f'{window_days}D')
        .sum()
        .reset_index()
        .rename(columns={'is_fraud': 'terminal_fraud_count_28d'})
    )

    # Merge back
    df.reset_index(inplace=True)
    df = df.merge(terminal_rolling, on=['TERMINAL_ID', 'TX_DATETIME'], how='left')

    return df


# def add_customer_avg_amount(df, window_days=14):
#     """
#     Adds rolling average transaction amount per customer over a time window.
#     """
#     df = df.sort_values('TX_DATETIME').copy()
#     df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

#     # Set up for merge
#     df['TRANSACTION_TIME'] = df['TX_DATETIME']
#     customer_df = df[['CUSTOMER_ID', 'TRANSACTION_ID', 'TX_AMOUNT', 'TRANSACTION_TIME']]

#     # Merge each transaction with past transactions of same customer
#     merged = customer_df.merge(customer_df, on='CUSTOMER_ID')
#     time_diff = (merged['TRANSACTION_TIME_x'] - merged['TRANSACTION_TIME_y']).dt.total_seconds() / (60 * 60 * 24)
#     merged = merged[(time_diff > 0) & (time_diff <= window_days)]

#     # Compute rolling average
#     avg_amounts = merged.groupby('TRANSACTION_ID_x')['TX_AMOUNT_y'].mean().rename('customer_avg_amount_14d')

#     # Merge back
#     df = df.merge(avg_amounts, left_on='TRANSACTION_ID', right_on='TRANSACTION_ID_x', how='left')
#     df['customer_avg_amount_14d'] = df['customer_avg_amount_14d'].fillna(0)

#     return df


def add_customer_avg_amount(df, window_days=14):
    """
    Adds rolling average transaction amount per customer over a time window.
    Uses groupby + rolling for speed.
    """
    df = df.sort_values('TX_DATETIME').copy()
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

    # Set datetime index for rolling
    df.set_index('TX_DATETIME', inplace=True)

    # Group by customer and apply rolling average
    rolling_avg = (
        df.groupby('CUSTOMER_ID')['TX_AMOUNT']
        .rolling(f'{window_days}D')
        .mean()
        .reset_index()
        .rename(columns={'TX_AMOUNT': 'customer_avg_amount_14d'})
    )

    # Merge back
    df.reset_index(inplace=True)
    df = df.merge(rolling_avg, on=['CUSTOMER_ID', 'TX_DATETIME'], how='left')
    df['customer_avg_amount_14d'] = df['customer_avg_amount_14d'].fillna(0)

    return df


def add_amount_deviation(df):
    """
    Adds a feature for deviation from customer average spend.
    """
    df['amount_deviation'] = df['TX_AMOUNT'] / (df['customer_avg_amount_14d'] + 1e-6)
    return df


