import os
import pandas as pd
import argparse
from collections import defaultdict


def preprocess_holdings(csv_path, threshold=3):
    """
    Mimics the structure of the A-LLMRec's group 'preprocess' function under SASREC.
    Instead of parsing a gzipped JSON file, we load a CSV file that contains
    holdings data. For each row (interaction), we count interactions per org
    and per stock, and then build a User dictionary mapping each org_id to a
    list of stock_ids in the order they appear (assuming row order is temporal).

    Parameters:
      csv_path (str): Path to the holdings CSV file.
      threshold (int): Minimum number of interactions for a user/stock to be included.
                       (Set to 1 if you want to include all interactions.)

    Returns:
      User (dict): Dictionary mapping org_id to list of stock_ids (in order).
      usermap (dict): Mapping from original org identifier to an integer ID.
      itemmap (dict): Mapping from original stock identifier to an integer ID.
    """
    # Initialize counters and maps.
    countU = defaultdict(int)
    countP = defaultdict(int)
    usermap = dict()
    itemmap = dict()
    User = dict()
    usernum = 0
    itemnum = 0

    # Load CSV into DataFrame.
    df = pd.read_csv(csv_path)

    # Iterate through the DataFrame rows in order.
    # (We assume the CSV's row order represents the interaction order.
    # BUG
    # THIS IS A CRITICAL MISS IN THE DATASET -- WHERE IS TIMESTAMP DATA THATS NOT ALL THE SAME??)
    for idx, row in df.iterrows():
        # For our context, assume the columns for user and item are:
        # 'org_id' and 'stock_id'
        org = row["org_id"]
        stock = row["stock_id"]

        # Increase counts (use thresholds if needed)
        countU[org] += 1
        countP[stock] += 1

        # Map org to a unique integer if not already mapped.
        if org not in usermap:
            usernum += 1
            usermap[org] = usernum
            User[usermap[org]] = []  # Use the mapped integer as the key.

        # Map stock to a unique integer if not already mapped.
        if stock not in itemmap:
            itemnum += 1
            itemmap[stock] = itemnum

        # Append the interaction.
        # Since no explicit timestamp exists, we simply record the mapped item id...
        User[usermap[org]].append(itemmap[stock])

    # Optionally, filter out users or items below a threshold.
    # (For example, only include users with at least 3 interactions.)
    User_filtered = {u: seq for u, seq in User.items() if len(seq) >= threshold}

    print(
        f"Total users (after filtering): {len(User_filtered)}; Total items: {len(itemmap)}"
    )
    return User_filtered, usermap, itemmap


def create_sequences_from_User(
    User, output_csv, user_key="org_id", item_key="stock_id"
):
    """
    Convert the User dictionary (from preprocess_holdings) into sequential training examples.

    For each user with at least two interactions, the history is all items except the last,
    and the target is the last item. The output CSV will have three columns:
      - org_id (mapped user id)
      - history: a space-separated list of item tokens
      - target: the next item to predict.

    Parameters:
      User (dict): Dictionary mapping user IDs to lists of item IDs.
      output_csv (str): Path to save the output CSV file.
      user_key (str): Column name for user ID in the output.
      item_key (str): Column name for items (for clarity, though items are stored as integers).

    Returns:
      df_seq (DataFrame): DataFrame of sequential examples.
    """
    data = []
    for user, sequence in User.items():
        if len(sequence) < 2:
            continue  # Skip users with fewer than 2 interactions.
        # Use all but the last item as history, and the last item as the target.
        history = " ".join(map(str, sequence[:-1]))
        target = sequence[-1]
        data.append({user_key: user, "history": history, "target": target})

    df_seq = pd.DataFrame(data)
    df_seq.to_csv(output_csv, index=False)
    return df_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess holdings CSV data to create sequential interaction examples for a SASRec-style model."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the holdings CSV file.",
        default="guilherme/data/raw/holdings_processed.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to the output CSV file with sequences.",
        default="guilherme/data/raw/"
    )
    args = parser.parse_args()

    # Preprocess the holdings CSV to get the User dictionary.
    User, usermap, itemmap = preprocess_holdings(args.input)

    # Create sequential training examples from the User dictionary.
    df_sequences = create_sequences_from_User(User, args.output)
    print("Sequential data created and saved to:", args.output)
