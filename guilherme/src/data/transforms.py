import pandas as pd
import numpy as np

def load_data(orgs_csv_path, holdings_csv_path):
    """
    Load the Orgs and Holdings CSV files.
    """
    df_orgs = pd.read_csv(orgs_csv_path)
    df_holdings = pd.read_csv(holdings_csv_path)
    
    # Convert the source_date column to datetime if it exists
    if 'source_date' in df_holdings.columns:
        df_holdings['source_date'] = pd.to_datetime(df_holdings['source_date'], errors='coerce')
    
    return df_orgs, df_holdings

def fill_avg_price(row, sector_medians):
    """
    Fill avg_price based on the following rules:
    1. If position_change_type is 'new' or 'soldall', return 0.
    2. If avg_price is available (not NaN), keep it.
    3. If avg_price is missing, fill with the median avg_price of the matching sector.
       If no median is available (or sector is missing), default to 0.
    """
    # Rule 1: Check position_change_type
    if row.get('position_change_type') in ['new', 'soldall']:
        return 0
    
    # Rule 2: If avg_price exists, return it
    if pd.notna(row.get('avg_price')):
        return row['avg_price']
    
    # Rule 3: Try to fill with the median for the matching sector
    sector = row.get('sector')
    if pd.notna(sector) and sector in sector_medians and pd.notna(sector_medians[sector]):
        return sector_medians[sector]
    
    # Default
    return 0

def fill_percent_fields(row):
    # Fill percent_change if missing and if both current and previous percentages are available
    if pd.isna(row.get('percent_change')):
        curr_pct = row.get('current_percent_of_portfolio')
        prev_pct = row.get('previous_percent_of_portfolio')
        if pd.notna(curr_pct) and pd.notna(prev_pct):
            row['percent_change'] = curr_pct - prev_pct
    
    # For percent_ownership, if position_change_type is 'new', set to 0.
    # Otherwise, if percent_ownership is missing, fill with current_percent_of_portfolio (if available).
    if row.get('position_change_type') == 'new':
        row['percent_ownership'] = 0
    else:
        if pd.isna(row.get('percent_ownership')):
            curr_pct = row.get('current_percent_of_portfolio')
            if pd.notna(curr_pct):
                row['percent_ownership'] = curr_pct
    
    return row

def fill_current_shares(row):
    # If current_shares is already present, leave it
    if pd.notna(row.get('current_shares')):
        return row['current_shares']
    
    # If both previous_shares and shares_change are available, compute current_shares
    if pd.notna(row.get('previous_shares')) and pd.notna(row.get('shares_change')):
        return row['previous_shares'] + row['shares_change']
    
    # Otherwise, return NaN (which means we'll drop this row later)
    return None

def impute_position_change_type(row):
    """
    Impute position_change_type based on current_shares, shares_change, ranking differences,
    and previous_percent_of_portfolio.

    Priority of signals:
    1. If current_shares is available and equals 0 and shares_change is negative, return 'soldall'.
    2. Otherwise, if shares_change is available:
         - If shares_change is positive and previous_percent_of_portfolio is 0, return 'new'.
         - Else if shares_change is positive, return 'addition'.
         - Else if shares_change is negative, return 'reduction'.
         - Else (shares_change is zero) return 'unchanged'.
    3. If shares_change is not available, use ranking differences:
         - If current_ranking < previous_ranking, return 'improved'.
         - If current_ranking > previous_ranking, return 'worsened'.
         - Otherwise, return 'unchanged'.
    4. If none of these signals are available, return 'unchanged'.
    """
    
    # Priority 1: Check if current_shares is available and equals 0 and shares_change is negative
    if pd.notna(row.get('current_shares')) and row['current_shares'] == 0:
        if pd.notna(row.get('shares_change')) and row['shares_change'] < 0:
            return 'soldall'
    
    # Optionally, if there's already a non-null value, keep it
    if pd.notna(row.get('position_change_type')):
        return row['position_change_type']
    
    # Priority 2: Use shares_change if available
    if pd.notna(row.get('shares_change')):
        if row['shares_change'] > 0:
            # Check for the 'new' condition: positive shares_change and previous_percent_of_portfolio == 0
            if pd.notna(row.get('previous_percent_of_portfolio')) and row['previous_percent_of_portfolio'] == 0:
                return 'new'
            else:
                return 'addition'
        elif row['shares_change'] < 0:
            return 'reduction'
        else:
            return 'unchanged'
    
    # Priority 3: Use ranking differences if available
    if pd.notna(row.get('current_ranking')) and pd.notna(row.get('previous_ranking')):
        if row['current_ranking'] < row['previous_ranking']:
            return 'improved'
        elif row['current_ranking'] > row['previous_ranking']:
            return 'worsened'
        else:
            return 'unchanged'
    
    # Fallback if no signals are available
    return 'unchanged'

def aggregate_investor_holdings(df_holdings):
    """
    Aggregate holdings data to derive investor-level features.
    For each investor (org_id), compute aggregated signals.
    """
    agg_funcs = {
        'shares_change': ['sum', 'mean'],
        'current_mv': ['sum', 'mean'],
        'previous_mv': ['sum', 'mean'],
        'current_percent_of_portfolio': ['mean'],
        'percent_ownership': ['mean'],
        'source_date': lambda x: (pd.Timestamp('today') - x.max()).days  # recency in days
    }
    
    investor_agg = df_holdings.groupby('org_id').agg(agg_funcs)
    # flatten multi-idx columns
    investor_agg.columns = ['_'.join([str(i) for i in col]).strip() for col in investor_agg.columns.values]
    investor_agg.reset_index(inplace=True)
    return investor_agg

def aggregate_company_holdings(df_holdings):
    """
    Aggregate holdings data to derive company-level features.
    For each company (using stock_id), compute aggregated signals.
    """
    agg_funcs = {
        'filer_id': 'count',  # number of investor records for this company
        'current_mv': ['sum', 'mean'],
        'current_percent_of_portfolio': ['mean'],
        'percent_ownership': ['mean']
    }
    
    company_agg = df_holdings.groupby('stock_id').agg(agg_funcs)
    company_agg.columns = ['_'.join([str(i) for i in col]).strip() for col in company_agg.columns.values]
    company_agg.reset_index(inplace=True)
    return company_agg

def engineer_investor_features(df_orgs, investor_agg):
    """
    Merge investor org information with aggregated holdings features
    and create additional engineered features.
    """
    # Filter investors (assuming org_type indicates 'investor')
    df_investors = df_orgs[df_orgs['org_type'].str.lower() == 'investor'].copy()
    
    # Merge aggregated holdings data
    investor_features = pd.merge(df_investors, investor_agg, on='org_id', how='left')
    
    # Example engineered feature: Investment Activity Score (adjust as needed)
    # To avoid division by zero, replace 0 recency with a small number (e.g., 1)
    recency = investor_features['source_date_<lambda>'].replace(0, 1)
    investor_features['investment_activity_score'] = investor_features['shares_change_sum'].fillna(0) / recency
    
    return investor_features

def engineer_company_features(df_orgs, company_agg):
    """
    Merge company org information with aggregated holdings features
    and create additional engineered features.
    """
    # Filter companies (assuming org_type indicates 'company')
    df_companies = df_orgs[df_orgs['org_type'].str.lower() == 'company'].copy()
    
    # Merge aggregated holdings data using ticker (assuming ticker corresponds to stock_id)
    company_features = pd.merge(df_companies, company_agg, left_on='ticker', right_on='stock_id', how='left')
    
    # Example engineered feature: Market Value Growth Ratio
    # This computes the ratio of the sum of current market value to the mean of previous market value.
    company_features['mv_growth_ratio'] = company_features['current_mv_sum'] / company_features['previous_mv_mean'].replace(0, np.nan)
    
    return company_features


if __name__ == "__main__":
    orgs_csv_path = '../../data/raw/holdings.csv'
    holdings_csv_path = '../../data/raw/orgs.csv'
    
    df_orgs, df_holdings = load_data(orgs_csv_path, holdings_csv_path)
    
    # Aggregate holdings for investors and companies
    investor_agg = aggregate_investor_holdings(df_holdings)
    company_agg = aggregate_company_holdings(df_holdings)
    
    # Engineer features for each tower
    investor_features = engineer_investor_features(df_orgs, investor_agg)
    company_features = engineer_company_features(df_orgs, company_agg)
    
    investor_features.to_csv('../../data/processed/investor_features.csv', index=False)
    company_features.to_csv('../../data/processed/company_features.csv', index=False)
    
    print("Feature transformation complete. Files saved in data/processed/.")
