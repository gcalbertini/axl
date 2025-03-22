import os
import pandas as pd
import numpy as np
import argparse
import base64
from IPython.display import HTML, display
import webbrowser

def load_data(orgs_csv_path, holdings_csv_path):
    """
    Load the Orgs and Holdings CSV files.
    """
    df_orgs = pd.read_csv(orgs_csv_path)
    df_holdings = pd.read_csv(holdings_csv_path)

    # Convert the source_date column to datetime if it exists
    if "source_date" in df_holdings.columns:
        df_holdings["source_date"] = pd.to_datetime(
            df_holdings["source_date"], errors="coerce"
        )

    return df_orgs, df_holdings


def populate_missing_fields_from_stocks(
    df_incoming, stocks_csv_path, merge_key="stock_id", mapping=None, default_value=0
):
    """
    Populate missing stocks-related values in the incoming DataFrame using reference data from stocks.csv.

    stocks.csv is expected to have columns:
        - id (which will be renamed to stock_id),
        - name (which will be renamed to stock_name),
        - symbol (which will be renamed to ticker),
        - sector,
    """
    if mapping is None:
        mapping = {
            "ticker": "symbol",
            "stock_name": "name",
            "sector": "sector",
            "industry": "industry",
        }

    # Determine which target columns are present in df_incoming.
    valid_mapping = {
        target: ref for target, ref in mapping.items() if target in df_incoming.columns
    }
    if not valid_mapping:
        return df_incoming

    # Save the original column order.
    original_cols = df_incoming.columns.tolist()

    # Load stocks.csv.
    df_stocks = pd.read_csv(stocks_csv_path)

    # Rename columns in stocks.csv: 'id' -> merge_key ('stock_id'), 'symbol' -> 'ticker', 'name' -> 'stock_name'
    if "id" in df_stocks.columns:
        df_stocks = df_stocks.rename(columns={"id": merge_key})
    # Optionally, you could also rename 'symbol' and 'name', but we handle that via mapping.

    # Subset stocks.csv to only include the merge key and the reference columns we need.
    required_ref_cols = list(set(valid_mapping.values()))
    required_cols = [merge_key] + required_ref_cols
    df_stocks_subset = df_stocks[required_cols].copy()

    # Merge the incoming DataFrame with the subset of stocks.
    df_merged = df_incoming.merge(
        df_stocks_subset, on=merge_key, how="left", suffixes=("", "_from_stocks")
    )

    # For each target field, fill missing values from the temporary reference column.
    for target_field, ref_field in valid_mapping.items():
        temp_col = f"{ref_field}_from_stocks"
        if temp_col in df_merged.columns:
            df_merged[target_field] = df_merged[target_field].fillna(
                df_merged[temp_col]
            )
            # Fill any remaining missing values with default_value.
            df_merged[target_field] = df_merged[target_field].fillna(default_value)
            df_merged.drop(columns=[temp_col], inplace=True)

    # Return only the original columns.
    df_result = df_merged[original_cols].copy()
    return df_result


def fill_avg_price(row, sector_medians):
    """
    Fill avg_price based on the following rules:
    1. If position_change_type is 'new' or 'soldall', return 0.
    2. If avg_price is available (not NaN), keep it.
    3. If avg_price is missing, fill with the median avg_price of the matching sector.
       If no median is available (or sector is missing), default to 0.
    """
    # Rule 1: Check position_change_type
    if row.get("position_change_type") in ["new", "soldall"]:
        return 0

    # Rule 2: If avg_price exists, return it
    if pd.notna(row.get("avg_price")):
        return row["avg_price"]

    # Rule 3: Try to fill with the median for the matching sector
    sector = row.get("sector")
    if (
        pd.notna(sector)
        and sector in sector_medians
        and pd.notna(sector_medians[sector])
    ):
        return sector_medians[sector]

    # Default
    return 0


def fill_percent_fields(row):
    # Fill percent_change if missing and if both current and previous percentages are available
    if pd.isna(row.get("percent_change")):
        curr_pct = row.get("current_percent_of_portfolio")
        prev_pct = row.get("previous_percent_of_portfolio")
        if pd.notna(curr_pct) and pd.notna(prev_pct):
            row["percent_change"] = curr_pct - prev_pct

    # For percent_ownership, if position_change_type is 'new', set to 0.
    # Otherwise, if percent_ownership is missing, fill with current_percent_of_portfolio (if available).
    if row.get("position_change_type") == "new":
        row["percent_ownership"] = 0
    else:
        if pd.isna(row.get("percent_ownership")):
            curr_pct = row.get("current_percent_of_portfolio")
            if pd.notna(curr_pct):
                row["percent_ownership"] = curr_pct

    return row


def fill_current_shares(row):
    # If current_shares is already present, leave it
    if pd.notna(row.get("current_shares")):
        return row["current_shares"]

    # If both previous_shares and shares_change are available, compute current_shares
    if pd.notna(row.get("previous_shares")) and pd.notna(row.get("shares_change")):
        return row["previous_shares"] + row["shares_change"]

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
    if pd.notna(row.get("current_shares")) and row["current_shares"] == 0:
        if pd.notna(row.get("shares_change")) and row["shares_change"] < 0:
            return "soldall"

    # Optionally, if there's already a non-null value, keep it
    if pd.notna(row.get("position_change_type")):
        return row["position_change_type"]

    # Priority 2: Use shares_change if available
    if pd.notna(row.get("shares_change")):
        if row["shares_change"] > 0:
            # Check for the 'new' condition: positive shares_change and previous_percent_of_portfolio == 0
            if (
                pd.notna(row.get("previous_percent_of_portfolio"))
                and row["previous_percent_of_portfolio"] == 0
            ):
                return "new"
            else:
                return "addition"
        elif row["shares_change"] < 0:
            return "reduction"
        else:
            return "unchanged"

    # Priority 3: Use ranking differences if available
    if pd.notna(row.get("current_ranking")) and pd.notna(row.get("previous_ranking")):
        if row["current_ranking"] < row["previous_ranking"]:
            return "improved"
        elif row["current_ranking"] > row["previous_ranking"]:
            return "worsened"
        else:
            return "unchanged"

    # Fallback if no signals are available
    return "unchanged"


def fill_missing_from_holdings_with_two_matches(
    df_orgs, df_holdings, key="org_id", target_fields=None
):
    """
    For each row in df_orgs, if any of the target fields [stock_id, stock_ticker, filer_id] is missing,
    use df_holdings to infer the missing value by requiring that at least two fields match.

    For each row in df_orgs (which has a unique org_id), do:
      1. Get all rows in df_holdings with the same org_id.
      2. Check which of the target fields (stock_id, stock_ticker, filer_id) are already present.
      3. If at least two fields are available, filter df_holdings for rows that match those available values.
      4. For each missing field, if the filtered subset yields exactly one unique candidate, use it;
         if multiple candidates exist, choose the mode; otherwise, leave it as missing.

    """
    if target_fields is None:
        target_fields = ["stock_id", "stock_ticker", "filer_id"]

    # Group df_holdings by org_id for fast lookup.
    grouped = df_holdings.groupby(key)

    def fill_row(row):
        org = row.get(key)
        if pd.isna(org) or org not in grouped.groups:
            return row
        # Get all rows for this org_id.
        subset = grouped.get_group(org)
        # Determine which target fields are available in the row.
        available = {
            field: row[field] for field in target_fields if pd.notna(row[field])
        }
        missing = [field for field in target_fields if pd.isna(row[field])]
        # Only proceed if at least two fields are available.
        if len(available) < 2:
            return row
        # Filter the subset based on available fields.
        filtered = subset.copy()
        for field, value in available.items():
            filtered = filtered[filtered[field] == value]
        # For each missing field, deduce a candidate.
        for field in missing:
            candidates = filtered[field].dropna().unique()
            if len(candidates) == 1:
                row[field] = candidates[0]
            elif len(candidates) > 1:
                mode_val = filtered[field].mode()
                if not mode_val.empty:
                    row[field] = mode_val.iloc[0]
        return row

    df_filled = df_orgs.apply(fill_row, axis=1)
    return df_filled


def engineer_holdings_features(df_holdings):
    """
    Add new columns to df_holdings:
      1. growth_of_mv
      2. days_since_source
      3. rank_change
      4. shares_change_ratio
      5. avg_share_price
    """
    # Small constant to avoid division by zero
    EPSILON = 1e-9

    # 1. growth_of_mv
    df_holdings["growth_of_mv"] = (
        df_holdings["current_mv"] - df_holdings["previous_mv"]
    ) / (df_holdings["previous_mv"] + EPSILON)

    # 2. days_since_source
    # Ensure source_date is a datetime; if it's already datetime, skip conversion
    if not np.issubdtype(df_holdings["source_date"].dtype, np.datetime64):
        df_holdings["source_date"] = pd.to_datetime(
            df_holdings["source_date"], errors="coerce"
        )
    df_holdings["days_since_source"] = (
        pd.Timestamp("now") - df_holdings["source_date"]
    ).dt.days

    # 3. rank_change
    df_holdings["rank_change"] = (
        df_holdings["previous_ranking"] - df_holdings["current_ranking"]
    )

    # 4. shares_change_ratio
    df_holdings["shares_change_ratio"] = df_holdings["shares_change"] / (
        df_holdings["previous_shares"] + EPSILON
    )

    # 5. avg_share_price
    df_holdings["avg_share_price"] = df_holdings["current_mv"] / (
        df_holdings["current_shares"] + EPSILON
    )
    # Just take medians for edge cases -- none found but just in case
    for col in df_holdings.columns:
        if df_holdings[col].isna().sum() > 0:
            median = df_holdings[col].median()
            df_holdings[col] = df_holdings[col].fillna(median)
    return df_holdings


def engineer_org_features(df_orgs, df_holdings):
    """
    For each org_id in df_orgs, derive:
      1. num_holdings
      2. total_mv
      3. avg_investment_size
      4. most_invested_sector
      5. diversification_score
    Then merge these features back into df_orgs.
    """
    # 1. num_holdings
    temp_num_holdings = (
        df_holdings.groupby("org_id")["stock_id"].nunique().reset_index()
    )
    temp_num_holdings.columns = ["org_id", "num_holdings"]

    # 2. total_mv
    temp_total_mv = df_holdings.groupby("org_id")["current_mv"].sum().reset_index()
    temp_total_mv.columns = ["org_id", "total_mv"]

    # 3. avg_investment_size
    temp_avg_investment = (
        df_holdings.groupby("org_id")["current_mv"].mean().reset_index()
    )
    temp_avg_investment.columns = ["org_id", "avg_investment_size"]

    # 4. most_invested_sector
    # First, sum the market value by (org_id, sector)
    temp_sector_mv = (
        df_holdings.groupby(["org_id", "sector"])["current_mv"].sum().reset_index()
    )
    # Then find the sector with the maximum sum for each org_id
    temp_sector_mv["rank"] = temp_sector_mv.groupby("org_id")["current_mv"].rank(
        method="first", ascending=False
    )
    # Keep only the top sector for each org_id
    temp_top_sector = temp_sector_mv.loc[
        temp_sector_mv["rank"] == 1, ["org_id", "sector"]
    ]
    temp_top_sector.columns = ["org_id", "most_invested_sector"]

    # 5. diversification_score
    # (Number of unique sectors in which the org invests)
    temp_diversification = (
        df_holdings.groupby("org_id")["sector"].nunique().reset_index()
    )
    temp_diversification.columns = ["org_id", "diversification_score"]

    # --- MERGE ALL TEMP DATAFRAMES ---
    # Start merging all partial results on org_id; if you used an inner join, you would
    # only retain org_id values that appear in every aggregation (e.g., only those with holdings in all of the aggregated dimensions).
    # Any org_id missing from one aggregator would be lost entirely. For instance, an organization might have no holdings in a particular
    #  sector (hence missing from temp_sector_mv), but it still has valid data in temp_num_holdings or temp_total_mv.
    # The outer join ensures you keep that organization’s row, just with NaN in the missing columns.
    df_features = pd.merge(temp_num_holdings, temp_total_mv, on="org_id", how="outer")
    df_features = pd.merge(df_features, temp_avg_investment, on="org_id", how="outer")
    df_features = pd.merge(df_features, temp_top_sector, on="org_id", how="outer")
    df_features = pd.merge(df_features, temp_diversification, on="org_id", how="outer")

    # Finally, merge these features into df_orgs
    df_orgs_enhanced = df_orgs.merge(df_features, on="org_id", how="left")

    # may have some edge cases but going with mode ~200
    for col in df_orgs_enhanced.columns:
        if df_orgs_enhanced[col].isna().sum() > 0:
            # Calculate mode; mode() returns a Series; we take the first element
            mode_value = df_orgs_enhanced[col].mode().iloc[0]
            df_orgs_enhanced[col] = df_orgs_enhanced[col].fillna(mode_value)
    return df_orgs_enhanced

def create_download_link(df, title="Download CSV file", filename="data.csv"):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode the CSV to Base64
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">{title}</a>'
    return href

def main(orgs_csv_path, holdings_csv_path, stocks_csv_path):
    import sys
    # Append parent directory for module lookup if needed
    sys.path.append(os.path.abspath(os.path.join("..")))

    # Load data
    df_orgs, df_holdings = load_data(orgs_csv_path, holdings_csv_path)
    df_orgs = df_orgs.rename(columns={"ticker": "stock_ticker"})

    # Clean and preprocess holdings data
    df_holdings = df_holdings.drop(
        columns=[
            "source",
            "quarter_id_owned",
            "quarter_end_price",
            "filer_street_address",
            "filer_city",
            "filer_state",
            "filer_zip_code",
        ]
    )  # no data or deeming unimportant

    df_holdings = populate_missing_fields_from_stocks(df_holdings, stocks_csv_path)
    df_holdings = df_holdings.dropna(
        subset=["stock_ticker"]
    )  # dropping rows with NaN stock_ticker

    # Fill missing values
    df_holdings[
        [
            "current_mv",
            "previous_mv",
            "previous_percent_of_portfolio",
            "current_percent_of_portfolio",
        ]
    ] = df_holdings[
        [
            "current_mv",
            "previous_mv",
            "previous_percent_of_portfolio",
            "current_percent_of_portfolio",
        ]
    ].fillna(
        0.0
    )

    df_holdings["current_ranking"] = df_holdings["current_ranking"].fillna(999999)
    df_holdings["previous_ranking"] = df_holdings["previous_ranking"].fillna(999999)
    df_holdings["filer_id"] = df_holdings["filer_id"].fillna(0)
    df_holdings["filer_name"] = df_holdings["filer_name"].fillna("unassigned filer")

    # Filter out unwanted stock tickers
    df_holdings = df_holdings[
        ~df_holdings["stock_ticker"].isin(
            ["0", "660", "0R87N:MEXI", "11G.OLD:FWB2", "9688.HK"]
        )
    ]

    # Fill missing shares and other fields
    df_holdings["previous_shares"] = df_holdings["previous_shares"].fillna(0)
    df_holdings["current_shares"] = df_holdings.apply(fill_current_shares, axis=1)
    df_holdings = df_holdings.dropna(subset=["current_shares"])
    df_holdings["position_change_type"] = df_holdings.apply(
        impute_position_change_type, axis=1
    )

    # Fill missing values for sector and industry
    df_holdings[["sector", "industry"]] = df_holdings[["sector", "industry"]].fillna(
        "Unknown"
    )
    df_holdings = df_holdings.apply(fill_percent_fields, axis=1)

    # Aggregate the median price for sectors
    sector_medians = df_holdings.groupby("sector")["avg_price"].median()
    df_holdings["avg_price"] = df_holdings.apply(
        lambda row: fill_avg_price(row, sector_medians), axis=1
    )

    df_holdings.info()

    # Now fill missing information in the organizations data
    df_orgs = fill_missing_from_holdings_with_two_matches(
        df_orgs, df_holdings
    )  # Big lift

    # Fill missing values in df_orgs
    df_orgs["stock_ticker"] = df_orgs["stock_ticker"].fillna("Unknown")
    df_orgs["stock_id"] = df_orgs["stock_id"].fillna(0)
    df_orgs = df_orgs.dropna(subset=["filer_id"])

    df_orgs = df_orgs.drop(columns=["cik"])
    df_orgs.info()

     # Create download links for the enhanced DataFrames
    orgs_html = create_download_link(df_orgs, "Download Org Features CSV", "org_features.csv")
    holdings_html = create_download_link(df_holdings, "Download Holdings Features CSV", "holdings_features.csv")
    
    # Write the HTML content to a temporary file and open it in the default web browser
    html_content = f"""
    <html>
      <head><title>CSV Download Links</title></head>
      <body>
        <h2>CSV Download Links</h2>
        <p>{orgs_html}</p>
        <p>{holdings_html}</p>
      </body>
    </html>
    """
    temp_html_path = "download_links.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Opening download links in your default web browser. If they do not appear, open {temp_html_path} manually.")
    webbrowser.open(f"file://{os.path.abspath(temp_html_path)}")

if __name__ == "__main__":
    # Use argparse to allow file paths to be passed from the command line.
    parser = argparse.ArgumentParser(
        description="Run data processing with external CSV files for stocks, holdings, and orgs."
    )
    parser.add_argument(
        "--orgs", type=str, required=False,
        help="Path to the orgs.csv file (e.g., C:/path/to/orgs.csv)",
        default="guilherme/data/raw/orgs.csv"
    )
    parser.add_argument(
        "--holdings", type=str, required=False,
        help="Path to the holdings.csv file (e.g., C:/path/to/holdings.csv)",
        default="guilherme/data/raw/holdings.csv"
    )
    parser.add_argument(
        "--stocks", type=str, required=False,
        help="Path to the stocks.csv file (e.g., C:/path/to/stocks.csv)",
        default="guilherme/data/raw/stocks.csv"
    )
    args = parser.parse_args()
    
    main(args.orgs, args.holdings, args.stocks)
