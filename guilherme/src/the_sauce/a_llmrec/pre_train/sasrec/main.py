from collections import defaultdict
import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gzip
from model import SASRec
from utils import (
    WarpSampler,
    data_partition,
    evaluate,
    evaluate_valid,
)

"""
NOTE GENERAL DISCUSSION:

In this assignment, the goal is to generate target lists—that is, to produce ranked recommendations of companies that customers 
(e.g., investment teams) should reach out to. Here's how this fits into our data:

    Investor Holdings as Implicit Behavior:
    The holdings table records past investments (or positions) by institutional investors. 
    This data reflects each investor's historical behavior and preferences—essentially, which companies they have invested in.

    Building Sequential Interaction Data:
    By grouping the holdings by org_id_encoded (investor) and ordering the interactions (even if only by row order -- see BUG below), 
    you create a sequence of investments for each investor. This sequence is used to train a sequential recommendation 
    model (like SASRec). The model learns the typical progression or patterns in an investor's portfolio.

    Generating Target Lists:
    Once trained, the model can predict the “next” company an investor might be interested in, based on their 
    past investment sequence. For each investor, the model outputs a ranked list of companies with associated scores (confidence levels).
    These ranked lists are the target lists that our group will provide to customers, advising them on which investors to reach out to regarding
    a specific company (e.g., cvshealth.com, baxter.com, netflix.com).

    Evaluation with HR and NDCG:
        HR@k: Measures whether the investor's actual next investment (or the most relevant company) appears in the top-k recommendations.
        NDCG@k: Evaluates not only if the target appears, but how high it is ranked in the list, providing a nuanced view of the quality of the ranking.

In Short
    Data Source: Holdings data capture past investor behavior.
    Processing: You build sequential interaction data (per investor) and train a model to predict the next investment.
    Output: The model produces a ranked list (target list) with confidence scores for each investor.
    Customer Value: These target lists guide customers on which investors might be receptive to outreach, based on historical behavior.

Thus, the recommendation process leverages our holdings data to generate actionable target lists for customers, and evaluation metrics like HR and NDCG help ensure these lists are both accurate and well-ranked.

"""

"""
NOTE TECHNICAL DISCUSSION:

NOTE: In a standard SASRec setup, the focus is on the interaction sequences 
derived from holdings. The orgs data (which might include bios and other dictdata) 
can later be used to enrich user representations in a hybrid or two-tower model 
(e.g., A-LLMRec). For SASRec alone, however, the sequential examples are the primary input.
MAJOR BUG Timestamp data is all wrong (all the same) so this is all assuming order inherent in CSV for transactions 

NOTE CONSIDER GIVING CANDIDATES TIMESTAMP DATA!!

In the context of this assignment, HR (Hit Ratio) and NDCG (Normalized Discounted Cumulative Gain)
 are used to evaluate how well our recommender system—based on 13D data—generates target lists 
 for companies with confidence scores for each investor.

    Hit Ratio (HR@k):
        Purpose: HR@k measures the proportion of investors for whom the actual target company (the one they eventually invested in)
          appears within the top-k recommendations.
        **If  system is tasked with generating target lists for companies (e.g., cvshealth.com, baxter.com, netflix.com), HR@k tells you whether the system
          is retrieving the correct companies among the top candidates for each investor. A high HR@k means
          that a large fraction of investors have the correct target company in their top-k list.

    Normalized Discounted Cumulative Gain (NDCG@k):
        Purpose: NDCG@k takes into account not only whether the target is 
        present but also its position in the recommendation list. It assigns higher scores when the target is ranked closer to the top.
        **In generating target lists with confidence scores, it's not enough just to include the right companies—the order matters.
          Investors are more likely to act on items that appear at the top of a ranked list. NDCG@k helps measure how well 
          our system ranks the most relevant companies. If the target company is placed high in the list, 
          it contributes to a higher NDCG score, indicating a more effective recommendation.


When you run the model (such as SASRec or a two-tower model like A-LLMRec), you can compute these metrics on the test or validation set. For each investor:

    HR@k: Check if the actual next investment (or the company we're targeting) is within the top-k recommendations.
    NDCG@k: Calculate a weighted gain based on the rank of the target company, normalized by the best possible ranking.

You can think of these as confidence scores in this recsys setting.

"""


def read_csv_with_progress(filename, chunksize=256, total_rows=None):
    """
    Read a CSV file in chunks, displaying progress with tqdm.

    Parameters:
      filename (str): The path to the CSV file.
      chunksize (int): The number of rows per chunk to read.
      total_rows (int, optional): Total number of rows in the file (if known) to show an accurate progress bar.

    Returns:
      DataFrame: The concatenated DataFrame containing all rows from the CSV.
    """
    chunks = []
    # If total_rows is provided, calculate total chunks
    if total_rows is not None:
        total_chunks = total_rows // chunksize + (
            1 if total_rows % chunksize != 0 else 0
        )
    else:
        total_chunks = None

    for chunk in tqdm(
        pd.read_csv(filename, chunksize=chunksize),
        total=total_chunks,
        desc="Reading CSV",
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df


def preprocess(df, processed_org_path, seq_out_file):
    """
    Process the holdings CSV file to generate:
      - Counts of interactions per investor and per company.
      - Mapping dictionaries (usermap and itemmap) to assign new integer IDs.
      - A User dictionary mapping each new investor ID to a list of interactions,
        where each interaction is represented as [timestamp, itemid].
        Since no timestamp is available, we assign one by incrementing a counter.
      - An org_dict dictionary mapping each new investor ID to dict information
        obtained from the orgs CSV (using the 'bio' field and optionally 'ticker').

    The function also saves:
      - org_dict as a gzipped pickle file.
      - A text file with one interaction per line in the format "userid itemid".

      User (dict): Mapping from new investor IDs to lists of interactions.
      usermap (dict): Mapping from original org_id to new integer IDs.
      itemmap (dict): Mapping from original stock_id to new integer IDs.
    """
    countU = defaultdict(int)
    countP = defaultdict(int)

    # First pass: count interactions.
    for idx, row in df.iterrows():
        org = row["org_id_encoded"]
        stock = row["stock_id"]
        countU[org] += 1
        countP[stock] += 1

    usermap = {}
    usernum = 0
    itemmap = {}
    itemnum = 0
    User = {}

    # Set threshold for minimum interactions.
    threshold = 4
    timestamp_counter = 0  # Arbitrary timestamp incrementer; assumed order inherently captures this dummy var

    # Second pass: build mappings and the User dictionary.
    for idx, row in df.iterrows():
        org = row["org_id_encoded"]
        stock = row["stock_id"]
        # Assign an arbitrary timestamp using the counter.
        timestamp = timestamp_counter
        timestamp_counter += 1

        if countU[org] < threshold or countP[stock] < threshold:
            continue

        # Map original org_id to new integer id.
        if org in usermap:
            userid = usermap[org]
        else:
            usernum += 1
            userid = usernum
            usermap[org] = userid
            User[userid] = []

        # Map original stock_id to new integer id.
        if stock in itemmap:
            itemid = itemmap[stock]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[stock] = itemid

        # Append interaction as [timestamp, itemid].
        User[userid].append([timestamp, itemid])

    # Load the orgs CSV to build org_dict.
    df_orgs = pd.read_csv(processed_org_path)
    """
    
    for orig_org, new_id in usermap.items():
        # Filter df_orgs rows where org_id matches.
        row = df_orgs[df_orgs["org_id_encoded"] == orig_org]
        if not row.empty:
            bio = row.iloc[0].get("bio", "No bio available")
            ticker = row.iloc[0].get("stock_ticker", "Unknown")
        else:
            bio = "No bio available"
            ticker = "Unknown"
        org_dict[new_id] = {"bio": bio, "stock_ticker": ticker}
    
    """
    # Create a dictionary with two keys: "bio" and "stock_ticker"
    text_name_dict = {"bio": {}, "stock_ticker": {}}
    for orig_org, new_id in usermap.items():
        # Filter df_orgs rows where org_id matches.
        row = df_orgs[df_orgs["org_id_encoded"] == orig_org]
        if not row.empty:
            bio = row.iloc[0].get("bio", "No bio available")
            ticker = row.iloc[0].get("stock_ticker", "Unknown")
        else:
            bio = "No bio available"
            ticker = "Unknown"
        # Instead of using new_id as the key for a nested dict,
        # assign it as a key in each top-level dictionary.
        text_name_dict["bio"][new_id] = bio
        text_name_dict["stock_ticker"][new_id] = ticker

    # Save the org_dict dictionary as a gzipped pickle file.
    org_dict_path = "guilherme/data/processed/holdings_org_dict.pkl.gz"
    with gzip.open(org_dict_path, "wb") as tf:
        pickle.dump(text_name_dict, tf)

    print("Total new org IDs:", usernum, "Total new stock IDs:", itemnum)

    # List of interactions for that user will be sorted in ascending order based on the timestamp.
    # This ensures that the interactions are ordered according to the time they occurred
    # (or, in our case, the arbitrary order defined by our arbitrary counter).
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    # Write out a text file with one interaction per line in the format "userid itemid".
    with open(seq_out_file, "w") as f:
        for user in User.keys():
            for interaction in User[user]:
                # interaction[1] is the itemid.
                f.write("%d %d\n" % (user, interaction[1]))


##############################################
# SASRec Main Routine
##############################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    help="Path to the processed holdings CSV file",
    default="guilherme/data/processed/holdings_processed.csv",
)
parser.add_argument(
    "--processed_orgs_csv",
    type=str,
    required=False,
    help="Path to the processed orgs CSV file",
    default="guilherme/data/processed/orgs_processed.csv",
)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=0.002, type=float)
parser.add_argument("--maxlen", default=50, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=160, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.5, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument(
    "--device",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    type=str,
)
parser.add_argument("--inference_only", default=False, action="store_true")
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument(
    "--seq_file",
    type=str,
    required=False,
    help="Path to output the raw interactions text file",
    default="guilherme/data/processed/sequences.txt",
)
# New argument: path to a processed holdings file with extra features.
extra_feature_cols = [
    "growth_of_mv",
    "rank_change",
    "shares_change_ratio",
    "avg_share_price",
    "percent_change",
    "avg_share_price",
    "position_change_type",
    "filer_id",
    "percent_ownership",
]
parser.add_argument(
    "--holdings_extra_features",
    type=str,
    required=False,
    help="Extra item features (e.g., from df_holdings_processed) to enrich vector in list.",
    default=extra_feature_cols,
)
args = parser.parse_args()

if __name__ == "__main__":
    # --- Preprocessing: Create raw interactions text file from holdings CSV ---
    if not os.path.exists(args.seq_file) or args.holdings_extra_features:
        print(
            "Detected new features to enrich baseline or raw interactions text file does not exist; creating one from holdings CSV..."
        )
        df = read_csv_with_progress(args.dataset, chunksize=256, total_rows=173922)
        print("Columns in the DataFrame:", df.columns.tolist())
        # Call preprocess() to generate interactions file and save org meta.
        # Assume preprocess() writes the file at args.seq_file.
        preprocess(df, args.processed_orgs_csv, args.seq_file)
        print("Raw interactions text file saved to", args.seq_file)
    else:
        print(
            "Raw interactions text file exists or no new features to enrich baseline; loading data from",
            args.seq_file,
        )

    # --- Partition the Interactions Data ---
    dataset = data_partition(args.seq_file)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print("Total users:", usernum, "Total items:", itemnum)
    num_batch = len(user_train) // args.batch_size
    total_length = sum(len(seq) for seq in user_train.values())
    print("Average sequence length: %.2f" % (total_length / len(user_train)))

    # --- Load extra item features ---

    # Hack -- lets include some one hot columns, too
    if args.holdings_extra_features:
        # Build a lookup dictionary: item_id -> extra features vector.
        extra_one_hot = [
            col
            for col in df.columns
            if col.startswith("sector_") or col.startswith("industry_")
        ]
        extra_feature_cols += extra_one_hot

        item_extra_features = {}
        for idx, row in df.iterrows():
            # Assuming "stock_id" here is already encoded as integer.
            item_id = row["stock_id"]
            features = row[extra_feature_cols].values.astype(np.float32)
            item_extra_features[item_id] = features
        # Determine the feature dimension.
        combined_feat_dim = len(extra_feature_cols)
    else:
        args.holdings_extra_features = None

    # --- Dataloader ---
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
    )

    # --- Model initialization ---
    # Here, we modify SASRec to accept additional item features.
    # We assume args now includes combined_feat_dim.
    args.combined_feat_dim = combined_feat_dim
    model = SASRec(usernum, itemnum, args, combined_feat_dim).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception as e:
            pass
    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(
                args.state_dict_path, map_location=torch.device(args.device)
            )
            kwargs["args"].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except Exception as e:
            print(
                "Failed loading state_dicts, please check file path:",
                args.state_dict_path,
            )
            import pdb

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print("Test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    """
        During training, both the positive interactions (what the user actually did) and
        negative samples (what the user did not do) are used. The model uses both sets of 
        logits to compute a loss (often a binary cross-entropy or pairwise ranking loss) that 
        encourages the model to rank positive items higher than negatives.
    
    """

    # --- Training Loop ---
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only:
            break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            # For the positive sequences, get the extra features:
            # Here we assume each sequence in 'pos' is of shape [seq_len] of item ids.
            # We'll build a tensor of shape [batch_size, seq_len, num_extra_features].
            batch_extra_feats = []
            for seq_item in pos:
                seq_feats = []
                for item in seq_item:
                    # Use a default vector if not found.
                    feat = item_extra_features.get(
                        item, np.zeros(len(extra_feature_cols), dtype=np.float32)
                    )
                    seq_feats.append(feat)
                batch_extra_feats.append(seq_feats)
            batch_extra_feats = torch.tensor(batch_extra_feats, dtype=torch.float).to(
                args.device
            )

            pos_logits, neg_logits = model(
                u, seq, pos, neg, item_side_features=batch_extra_feats
            )
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

            if step % 100 == 0:
                print("Epoch {} Step {} Loss: {:.4f}".format(epoch, step, loss.item()))

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating...")
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(
                "\nEpoch: {}, Time: {:.2f}s, Valid (NDCG@10: {:.4f}, HR@10: {:.4f}), Test (NDCG@10: {:.4f}, HR@10: {:.4f})".format(
                    epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]
                )
            )
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = "guilherme/experiments/"
            fname = (
                "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth".format(
                    args.num_epochs,
                    args.lr,
                    args.num_blocks,
                    args.num_heads,
                    args.hidden_units,
                    args.maxlen,
                )
            )
            os.makedirs(folder, exist_ok=True)
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))

    sampler.close()
    print("Training completed.")
