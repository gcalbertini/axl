import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import base64
import webbrowser
from IPython.display import HTML
from tqdm import tqdm
import pickle


##############################################
# DATA LOADING & MISSING FIELD POPULATION
##############################################
def load_data(orgs_csv_path, holdings_csv_path):
    """
    Load the orgs and holdings CSV files.

    Parameters:
      orgs_csv_path (str): Path to the orgs CSV file.
      holdings_csv_path (str): Path to the holdings CSV file.

    Returns:
      df_orgs (DataFrame): DataFrame containing organization data.
      df_holdings (DataFrame): DataFrame containing holdings data.

    """
    df_orgs = pd.read_csv(orgs_csv_path)
    df_holdings = pd.read_csv(holdings_csv_path)

    # Convert 'source_date' to datetime if present.
    if "source_date" in df_holdings.columns:
        df_holdings["source_date"] = pd.to_datetime(
            df_holdings["source_date"], errors="coerce"
        )
    return df_orgs, df_holdings


def impute_position_change_type(row):
    if pd.notna(row.get("current_shares")) and row["current_shares"] == 0:
        if pd.notna(row.get("shares_change")) and row["shares_change"] < 0:
            return "soldall"
    if pd.notna(row.get("position_change_type")):
        return row["position_change_type"]
    if pd.notna(row.get("shares_change")):
        if row["shares_change"] > 0:
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
    if pd.notna(row.get("current_ranking")) and pd.notna(row.get("previous_ranking")):
        if row["current_ranking"] < row["previous_ranking"]:
            return "improved"
        elif row["current_ranking"] > row["previous_ranking"]:
            return "worsened"
        else:
            return "unchanged"
    return "unchanged"


def fill_percent_fields(row):
    if pd.isna(row.get("percent_change")):
        curr_pct = row.get("current_percent_of_portfolio")
        prev_pct = row.get("previous_percent_of_portfolio")
        if pd.notna(curr_pct) and pd.notna(prev_pct):
            row["percent_change"] = curr_pct - prev_pct
    if row.get("position_change_type") == "new":
        row["percent_ownership"] = 0
    else:
        if pd.isna(row.get("percent_ownership")):
            curr_pct = row.get("current_percent_of_portfolio")
            if pd.notna(curr_pct):
                row["percent_ownership"] = curr_pct
    return row


def fill_avg_price(row, sector_medians):
    EPSILON = 1e-9
    if row.get("position_change_type") in ["new", "soldall"]:
        return 0
    if pd.notna(row.get("avg_price")):
        return row["avg_price"]
    sector = row.get("sector")
    if (
        pd.notna(sector)
        and sector in sector_medians
        and pd.notna(sector_medians[sector])
    ):
        return sector_medians[sector]
    return 0


def fill_current_shares(row):
    if pd.notna(row.get("current_shares")):
        return row["current_shares"]
    if pd.notna(row.get("previous_shares")) and pd.notna(row.get("shares_change")):
        return row["previous_shares"] + row["shares_change"]
    return None


def fill_missing_from_holdings_with_two_matches(
    df_orgs, df_holdings, key="org_id", target_fields=None
):
    if target_fields is None:
        target_fields = ["stock_id", "filer_id", "stock_ticker"]
    grouped = df_holdings.groupby(key)

    def fill_row(row):
        org = row.get(key)
        if pd.isna(org) or org not in grouped.groups:
            return row
        subset = grouped.get_group(org)
        available = {
            field: row[field] for field in target_fields if pd.notna(row[field])
        }
        missing = [field for field in target_fields if pd.isna(row[field])]
        if len(available) < 2:
            return row
        filtered = subset.copy()
        for field, value in available.items():
            filtered = filtered[filtered[field] == value]
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


def convert_key(val):
    """
    Convert a key value to a standardized string.
    For numeric values, convert to float then int to remove trailing decimals,
    otherwise return the stripped string.
    """
    try:
        return str(int(float(val)))
    except Exception:
        return str(val).strip()


def convert_key(val):
    """
    Convert a key value to a standardized string.
    For numeric values, convert to float then int to remove trailing decimals,
    otherwise return the stripped string.
    """
    try:
        return str(int(float(val)))
    except Exception:
        return str(val).strip()


def populate_missing_stock_fields_in_holdings(df_holdings, stocks_csv_path):
    """
    Populate missing values in df_holdings for 'stock_ticker' and 'stock_id'
    using reference data from stocks.csv.

    stocks.csv is expected to contain:
      - 'id': the stock identifier (to be renamed to 'stock_id'),
      - 'symbol': the stock ticker (to be renamed to 'stock_ticker'),
      - (other columns are ignored).

    For each row in df_holdings:
      - If stock_id is present but stock_ticker is missing, look up the stock_ticker
        from stocks.csv using stock_id.
      - If stock_ticker is present but stock_id is missing, look up the stock_id using stock_ticker.
      - If both are missing or if a lookup fails, the row is marked for deletion.
      - In addition, rows with a stock_id equal to -1 or 0 (after conversion) are dropped.

    Returns:
      Updated df_holdings with missing fields filled from stocks.csv, and with rows having
      invalid stock_id (-1 or 0) removed.
    """
    # Read only the necessary columns from stocks.csv.
    df_stocks = pd.read_csv(stocks_csv_path, usecols=['id', 'symbol'])
    
    # Rename columns to match df_holdings.
    df_stocks.rename(columns={'id': 'stock_id', 'symbol': 'stock_ticker'}, inplace=True)
    
    # Standardize the key fields to strings.
    df_stocks['stock_id'] = df_stocks['stock_id'].astype(str).str.strip()
    df_stocks['stock_ticker'] = df_stocks['stock_ticker'].astype(str).str.strip()
    
    # Build lookup dictionaries.
    lookup_by_id = df_stocks.set_index('stock_id')['stock_ticker'].to_dict()
    lookup_by_ticker = df_stocks.set_index('stock_ticker')['stock_id'].to_dict()
    
    # Prepare a list to record rows to drop.
    drop_indices = []
    
    def update_row(row):
        # Standardize current row values.
        stock_id = str(row.get('stock_id')).strip() if pd.notna(row.get('stock_id')) else None
        stock_ticker = str(row.get('stock_ticker')).strip() if pd.notna(row.get('stock_ticker')) and row.get('stock_ticker') != "" else None
        
        # Case 1: Both values exist -> do nothing.
        if stock_id and stock_ticker:
            return row
        # Case 2: stock_id exists but stock_ticker is missing -> look up ticker.
        elif stock_id and not stock_ticker:
            candidate = lookup_by_id.get(stock_id)
            if candidate:
                row['stock_ticker'] = candidate
            else:
                drop_indices.append(row.name)
        # Case 3: stock_ticker exists but stock_id is missing -> look up stock_id.
        elif stock_ticker and not stock_id:
            candidate = lookup_by_ticker.get(stock_ticker)
            if candidate:
                row['stock_id'] = candidate
            else:
                drop_indices.append(row.name)
        # Case 4: Both are missing -> drop the row.
        else:
            drop_indices.append(row.name)
        return row

    df_holdings_updated = df_holdings.apply(update_row, axis=1)
    df_holdings_updated = df_holdings_updated.drop(index=drop_indices)
    
    # Drop rows where stock_id (converted to float) is -1 or 0.
    def valid_stock_id(sid):
        try:
            return float(sid) not in [-1, 0]
        except Exception:
            return False

    df_holdings_updated = df_holdings_updated[df_holdings_updated['stock_id'].apply(valid_stock_id)]
    
    return df_holdings_updated

##############################################
# FEATURE ENGINEERING FUNCTIONS
##############################################
def engineer_holdings_features(df_holdings):
    """
    Add new columns to df_holdings for downstream feature engineering.

    New features added:
      1. growth_of_mv: Percentage growth (or decline) in market value.
      2. rank_change: Change in ranking (previous_ranking - current_ranking).
      3. shares_change_ratio: Relative change in shares (shares_change / previous_shares).
      4. avg_share_price: Approximate average share price (current_mv / current_shares).

    Also fills any remaining missing values with the median of each column.
    """
    EPSILON = 1e-9  # Small constant to prevent division by zero.

    # 1. Compute growth_of_mv
    df_holdings["growth_of_mv"] = (
        df_holdings["current_mv"] - df_holdings["previous_mv"]
    ) / (df_holdings["previous_mv"] + EPSILON)

    # 2. Compute rank_change: the difference between previous and current ranking.
    df_holdings["rank_change"] = (
        df_holdings["previous_ranking"] - df_holdings["current_ranking"]
    )

    # 3. Compute shares_change_ratio.
    df_holdings["shares_change_ratio"] = df_holdings["shares_change"] / (
        df_holdings["previous_shares"] + EPSILON
    )

    # 4. Compute avg_share_price.
    df_holdings["avg_share_price"] = df_holdings["current_mv"] / (
        df_holdings["current_shares"] + EPSILON
    )

    # Fill any remaining missing values in each column with the column's median.
    for col in df_holdings.columns:
        if df_holdings[col].isna().sum() > 0:
            median = df_holdings[col].median()
            df_holdings[col] = df_holdings[col].fillna(median)

    return df_holdings


def engineer_org_features(df_orgs, df_holdings):
    """
    Generate and merge additional features into df_orgs based on aggregated holdings data.

    New features generated for each org_id:
      1. num_holdings: Number of unique stock_ids held.
      2. total_mv: Sum of current market values.
      3. avg_investment_size: Mean current market value per holding.
      4. most_invested_sector: The sector where the org has invested the most.
      5. diversification_score: Number of unique sectors the org invests in.

    These aggregated features are merged back into df_orgs.
    Missing values in the resulting DataFrame are filled with the mode of each column.
    """
    # 1. Number of unique holdings per organization.
    temp_num_holdings = (
        df_holdings.groupby("org_id")["stock_id"].nunique().reset_index()
    )
    temp_num_holdings.columns = ["org_id", "num_holdings"]

    # 2. Total market value invested per organization.
    temp_total_mv = df_holdings.groupby("org_id")["current_mv"].sum().reset_index()
    temp_total_mv.columns = ["org_id", "total_mv"]

    # 3. Average investment size per organization.
    temp_avg_investment = (
        df_holdings.groupby("org_id")["current_mv"].mean().reset_index()
    )
    temp_avg_investment.columns = ["org_id", "avg_investment_size"]

    # 4. Most invested sector per organization.
    # Sum current_mv by org_id and sector.
    temp_sector_mv = (
        df_holdings.groupby(["org_id", "sector"])["current_mv"].sum().reset_index()
    )
    # Rank sectors per org by market value in descending order.
    temp_sector_mv["rank"] = temp_sector_mv.groupby("org_id")["current_mv"].rank(
        method="first", ascending=False
    )
    # Keep the top sector for each org.
    temp_top_sector = temp_sector_mv.loc[
        temp_sector_mv["rank"] == 1, ["org_id", "sector"]
    ]
    temp_top_sector.columns = ["org_id", "most_invested_sector"]

    # 5. Diversification score: count of unique sectors.
    temp_diversification = (
        df_holdings.groupby("org_id")["sector"].nunique().reset_index()
    )
    temp_diversification.columns = ["org_id", "diversification_score"]

    # Merge all aggregated features using outer joins to preserve all org_ids.
    df_features = pd.merge(temp_num_holdings, temp_total_mv, on="org_id", how="outer")
    df_features = pd.merge(df_features, temp_avg_investment, on="org_id", how="outer")
    df_features = pd.merge(df_features, temp_top_sector, on="org_id", how="outer")
    df_features = pd.merge(df_features, temp_diversification, on="org_id", how="outer")

    # Merge these aggregated features back into df_orgs.
    df_orgs_enhanced = df_orgs.merge(df_features, on="org_id", how="left")

    # For any missing values in the merged features, fill with the mode.
    for col in df_orgs_enhanced.columns:
        if df_orgs_enhanced[col].isna().sum() > 0:
            mode_value = df_orgs_enhanced[col].mode().iloc[0]
            df_orgs_enhanced[col] = df_orgs_enhanced[col].fillna(mode_value)
    return df_orgs_enhanced


##############################################
# ENCODING & SCALING FUNCTIONS
##############################################
def process_csv(df, onehot_cols, ordinal_cols, numeric_override=None, key_cols=None):
    """
    Process a DataFrame by:
      1. Applying ordinal encoding on specified columns.
      2. One-hot encoding specified nominal columns.
      3. Scaling remaining numeric columns.

    Parameters:
      df (DataFrame): Input DataFrame.
      onehot_cols (list): Columns to one-hot encode.
      ordinal_cols (list): Columns to ordinally encode.
      numeric_override (list or None): If provided, only these columns are scaled.
      key_cols (list or None): Columns that should remain unchanged (not encoded or scaled).

    Returns:
      df (DataFrame): Processed DataFrame.

    Note:
      - Key columns (e.g. org_id, email_extension) are preserved.
      - Shared categorical columns (stock_id, filer_id, stock_ticker) are one-hot encoded.
    """
    # Temporarily remove key columns.
    keys = {}
    if key_cols is not None:
        for col in key_cols:
            if col in df.columns:
                keys[col] = df[col]
                df = df.drop(columns=[col])

    # ----- Ordinal Encoding -----
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0]
        else:
            print(f"Warning: Ordinal column '{col}' not found.")

    # ----- One-Hot Encoding -----
    # For each specified column, convert to dummy variables.
    for col in onehot_cols:
        if col in df.columns:
            # If we want to enforce consistent categories for shared columns,
            # the column should already be a pd.Categorical with defined categories.
            dummies = pd.get_dummies(df[col], prefix=col)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
        else:
            print(f"Warning: One-hot column '{col}' not found.")

    # ----- Scaling Numeric Columns -----
    all_numeric = df.select_dtypes(include=["number"]).columns.tolist()
    dummy_cols = []
    for col in onehot_cols:
        dummy_cols.extend([c for c in df.columns if c.startswith(f"{col}_")])

    if numeric_override is not None:
        scale_cols = [col for col in numeric_override if col in df.columns]
    else:
        scale_cols = [
            col
            for col in all_numeric
            if col not in ordinal_cols and col not in dummy_cols
        ]

    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        print(f"Scaled columns: {scale_cols}")
    else:
        print("No numeric columns to scale.")

    # Reattach key columns.
    if key_cols is not None:
        for col, series in keys.items():
            df[col] = series
        # Optionally reorder to have key_cols first.
        remaining_cols = [col for col in df.columns if col not in key_cols]
        df = df[key_cols + remaining_cols]

    return df


def create_download_link(df, title="Download CSV file", filename="data.csv"):
    """
    Convert the DataFrame to a CSV string, encode it in Base64,
    and return an HTML anchor tag for downloading the file.

    Parameters:
      df (DataFrame): DataFrame to convert.
      title (str): Text to display in the link.
      filename (str): Filename for the download.

    Returns:
      str: HTML download link.
    """
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">{title}</a>'
    return href


def generate_html_file(links_dict, html_filename="download_links.html"):
    """
    Generate an HTML file containing the download links.

    Parameters:
      links_dict (dict): Dictionary where keys are link descriptions and values are HTML link strings.
      html_filename (str): Filename for the generated HTML file.

    Returns:
      str: Absolute path to the generated HTML file.
    """
    html_links = "".join(f"<p>{link}</p>" for link in links_dict.values())
    html_content = f"""
    <html>
      <head><title>Download Processed CSV Files</title></head>
      <body>
        <h2>Download Processed CSV Files</h2>
        {html_links}
      </body>
    </html>
    """
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML file with download links saved to {html_filename}")
    return os.path.abspath(html_filename)


def to_csv_with_progress(df, filename, chunksize=256):
    """
    Write DataFrame to CSV in chunks, displaying progress with tqdm.

    Parameters:
      df (DataFrame): The DataFrame to write.
      filename (str): Output CSV file path.
      chunksize (int): Number of rows to write per chunk.
    """
    n = len(df)
    print(f"df for {filename} has {n} rows")
    with open(filename, "w", encoding="utf-8", newline="") as f:
        # Write header once.
        df.head(0).to_csv(f, index=False)
        # Write the DataFrame in chunks.
        for i in tqdm(range(0, n, chunksize), desc="Writing CSV"):
            df.iloc[i : i + chunksize].to_csv(f, index=False, header=False)


def create_union_mapping(df1, df2, col):
    """
    Create a mapping dictionary for a column based on the union of unique values from two DataFrames.

    Parameters:
    df1, df2 (DataFrame): The DataFrames to consider.
    col (str): The column name to compute the union for.

    Returns:
    mapping (dict): A dictionary mapping each unique value to a unique integer.
    """
    union_values = sorted(
        set(df1[col].dropna().unique()).union(set(df2[col].dropna().unique()))
    )
    mapping = {val: idx for idx, val in enumerate(union_values)}
    return mapping


##############################################
# MAIN FUNCTION
##############################################
def main(orgs_csv_path, holdings_csv_path, stocks_csv_path):
    """
    Main function to process orgs, holdings, and stocks CSV files.

    Steps:
      1. Load raw CSV files.
      2. Preprocess holdings:
         - Drop unneeded columns.
         - Populate missing fields using stocks.csv.
         - Fill missing market values, ranking, and shares.
         - Drop rows missing current_shares.
         - Impute position_change_type.
         - Fill missing sector and industry.
         - Fill percent fields and compute engineered features.
      3. Feature Engineering on Holdings.
      4. Preprocess orgs:
         - Fill missing fields using holdings data.
         - Drop unnecessary columns.
      5. Feature Engineering on Orgs.
      6. Consistent Encoding & Scaling:
         - The key columns that remain unchanged are org_id and email_extension.
      7. Generate HTML download links for the processed CSV files and open them in the default browser.
    """
    import sys

    sys.path.append(os.path.abspath(os.path.join("..")))

    # Load raw CSV files.
    df_orgs, df_holdings = load_data(orgs_csv_path, holdings_csv_path)
    df_orgs = df_orgs.rename(columns={"ticker": "stock_ticker"})  # For consistency.

    # --- Preprocess Holdings Data ---
    # Drop columns that are not considered for now.
    df_holdings = df_holdings.drop(
        columns=[
            "source",
            "quarter_id_owned",
            "quarter_end_price",
            "filer_street_address",
            "filer_city",
            "filer_state",
            "filer_zip_code",
            "id",
            "source_date",
        ]
    )
    # Populate missing fields using stocks.csv, namely stock_ticker and stock_id else have to drop if we can't find
    # BUG 429 NaNs here from ingestion, 429 and expect to degrade performance
    df_holdings = populate_missing_stock_fields_in_holdings(
        df_holdings, stocks_csv_path
    )

    # Fill missing market values and percentage fields with 0.
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

    # Fill missing ranking fields with a default high value.
    df_holdings["current_ranking"] = df_holdings["current_ranking"].fillna(999999)
    df_holdings["previous_ranking"] = df_holdings["previous_ranking"].fillna(999999)

    # Fill missing filer information.
    df_holdings["filer_id"] = df_holdings["filer_id"].fillna(0)

    # Drop columns that are not required -- already have id accounted for right before
    df_holdings = df_holdings.drop(
        columns=[
            "filer_name",
        ]
    )

    # Remove unwanted stock_ticker values.
    df_holdings = df_holdings[
        ~df_holdings["stock_ticker"].isin(
            ["0", "660", "0R87N:MEXI", "11G.OLD:FWB2", "9688.HK"]
        )
    ]

    # Fill missing previous_shares and compute current_shares if missing.
    df_holdings["previous_shares"] = df_holdings["previous_shares"].fillna(0)
    df_holdings["current_shares"] = df_holdings.apply(
        lambda row: (
            row["previous_shares"] + row["shares_change"]
            if pd.isna(row.get("current_shares"))
            and pd.notna(row.get("previous_shares"))
            and pd.notna(row.get("shares_change"))
            else row.get("current_shares")
        ),
        axis=1,
    )
    df_holdings = df_holdings.dropna(subset=["current_shares"])

    # Impute position_change_type using a custom function.
    df_holdings["position_change_type"] = df_holdings.apply(
        impute_position_change_type, axis=1
    )

    # Fill missing 'sector' and 'industry' with "Unknown".
    df_holdings[["sector", "industry"]] = df_holdings[["sector", "industry"]].fillna(
        "Unknown"
    )

    # Fill percent-based fields.
    df_holdings = df_holdings.apply(fill_percent_fields, axis=1)

    # Compute average price using median values per sector.
    sector_medians = df_holdings.groupby("sector")["avg_price"].median()
    df_holdings["avg_price"] = df_holdings.apply(
        lambda row: fill_avg_price(row, sector_medians), axis=1
    )

    df_holdings = df_holdings.drop(
        columns=["stock_name"]
    )  # stock_id, stock_ticker describes these


    # --- Feature Engineering on Holdings ---
    df_holdings = engineer_holdings_features(df_holdings)

    # --- Preprocess Orgs Data ---
    df_orgs = fill_missing_from_holdings_with_two_matches(df_orgs, df_holdings)
    df_orgs = df_orgs.dropna(subset=["filer_id", "stock_id", "stock_ticker"])
    df_orgs = df_orgs.drop(columns=["cik"])

    # --- Feature Engineering on Orgs ---
    df_orgs = engineer_org_features(df_orgs, df_holdings)

    ##############################################
    # ENSURE CONSISTENT ONE-HOT ENCODING FOR SHARED COLUMNS
    ##############################################
    # Shared columns (stock id and filer_id) should be label-encoded (not one-hot encoded) along with keys (email and org id)
    # These are strings that can be encoded as int to save space and work with downstream modeling (expect int ref).
    # We also assume filer and stock ids are matching as they come from WhaleWatchers API which we assume is consistent.
    org_id_map = create_union_mapping(df_orgs, df_holdings, "org_id")
    email_ext_map = create_union_mapping(df_orgs, df_holdings, "email_extension")

    # Apply the mappings to both DataFrames, creating new encoded columns.
    df_orgs["org_id_encoded"] = df_orgs["org_id"].map(org_id_map)
    df_holdings["org_id_encoded"] = df_holdings["org_id"].map(org_id_map)

    df_orgs["email_extension_encoded"] = df_orgs["email_extension"].map(email_ext_map)
    df_holdings["email_extension_encoded"] = df_holdings["email_extension"].map(
        email_ext_map
    )

    df_holdings = df_holdings.drop(columns=["org_id", "email_extension"])
    df_orgs = df_orgs.drop(columns=["org_id", "email_extension"])

    # BUG at this point have

    # Save the mapping dictionaries to files in your raw folder.
    with open("guilherme/data/raw/org_id_map.pkl", "wb") as f:
        pickle.dump(org_id_map, f)
    with open("guilherme/data/raw/email_extension_map.pkl", "wb") as f:
        pickle.dump(email_ext_map, f)

    ##############################################
    # CUSTOM ENCODING & SCALING
    ##############################################
    # Define key columns that should remain unchanged after encoding them; stock_ticker and filer_id passed in for ease of reference downstream despite char
    key_cols_passed = [
        "org_id_encoded",
        "email_extension_encoded",
        "stock_ticker",
        "filer_id",
    ]

    # For Holdings:
    # Nominal columns to one-hot encode
    onehot_cols_holdings = ["security_type", "sector", "industry"]
    # Ordinal encoding for ranking columns.
    ordinal_cols_holdings = [
        "current_ranking",
        "previous_ranking",
        "position_change_type",
    ]
    # Numeric columns to scale (engineered and original numeric features, excluding keys and the shared categorical columns which are now encoded).
    numeric_cols_holdings = [
        "shares_change",
        "current_percent_of_portfolio",
        "previous_percent_of_portfolio",
        "current_mv",
        "previous_mv",
        "current_shares",
        "previous_shares",
        "percent_ownership",
        "avg_price",
        "percent_change",
        "growth_of_mv",
        "rank_change",
        "shares_change_ratio",
        "avg_share_price",
    ]
    df_holdings_processed = process_csv(
        df_holdings.copy(),
        onehot_cols=onehot_cols_holdings,
        ordinal_cols=ordinal_cols_holdings,
        numeric_override=numeric_cols_holdings,
        key_cols=key_cols_passed,
    )
    holdings_output = "guilherme/data/processed/holdings_processed.csv"
    print("Going to take a while to scale holdings...")
    to_csv_with_progress(
        df_holdings_processed, holdings_output
    )  # THIS MAY BE GINORMOUS
    print("Scaled holdings and CSV writing completed.")

    # For Orgs:
    onehot_cols_org = ["org_type", "most_invested_sector"]
    ordinal_cols_org = []  # No ordinal encoding for orgs.
    numeric_cols_org = [
        "num_holdings",
        "total_mv",
        "avg_investment_size",
        "diversification_score",
    ]
    df_orgs_processed = process_csv(
        df_orgs.copy(),
        onehot_cols=onehot_cols_org,
        ordinal_cols=ordinal_cols_org,
        numeric_override=numeric_cols_org,
        key_cols=key_cols_passed,
    )

    # bio is text

    orgs_output = "guilherme/data/processed/orgs_processed.csv"
    to_csv_with_progress(df_orgs_processed, orgs_output)

    ##############################################
    # GENERATE HTML DOWNLOAD LINKS
    ##############################################
    orgs_html = create_download_link(
        df_orgs_processed, "Download Processed Org Features CSV", "org_features.csv"
    )
    holdings_html = create_download_link(
        df_holdings_processed,
        "Download Processed Holdings Features CSV",
        "holdings_features.csv",
    )

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

    print(
        f"Opening download links in your default web browser. If they do not appear, open {temp_html_path} manually."
    )
    webbrowser.open(f"file://{os.path.abspath(temp_html_path)}")


##############################################
# SCRIPT ENTRY POINT
##############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process orgs, holdings, and stocks CSV files with feature engineering and consistent scaling for shared columns, then generate HTML download links."
    )
    parser.add_argument(
        "--orgs",
        type=str,
        required=False,
        help="Path to the orgs.csv file (default: guilherme/data/raw/orgs.csv)",
        default="guilherme/data/raw/orgs.csv",
    )
    parser.add_argument(
        "--holdings",
        type=str,
        required=False,
        help="Path to the holdings.csv file (default: guilherme/data/raw/holdings.csv)",
        default="guilherme/data/raw/holdings.csv",
    )
    parser.add_argument(
        "--stocks",
        type=str,
        required=False,
        help="Path to the stocks.csv file (default: guilherme/data/raw/stocks.csv)",
        default="guilherme/data/raw/stocks.csv",
    )
    args = parser.parse_args()

    main(args.orgs, args.holdings, args.stocks)
