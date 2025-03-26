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

In this assignment, the objective is to generate target lists—that is, to produce ranked recommendations 
of investors who are likely to invest in a specific company (represented by its stock ticker). 

    Investor Holdings as Implicit Behavior:
    The holdings table records past investments (or positions) made by institutional investors. This data 
    reflects each investor's historical behavior and preferences—specifically, which companies they have 
    invested in.

    Building Sequential Interaction Data:
    By grouping the holdings by investor (org_id_encoded) and ordering their transactions (using row order, 
    as the timestamp data is currently not available), you can create a sequence of investments for each investor. 
    This sequence is used to train a sequential recommendation model (like SASRec). The model learns the typical 
    progression or patterns in an investor's portfolio.

    Generating Target Lists:
    Once trained, the OG A-LLMRec model predicts the “next” company each investor might invest in, based solely on their 
    historical investment sequence. For our case, given target company (e.g., cvshealth.com, baxter.com, netflix.com), 
    the model outputs a ranked list of investors, each with an associated confidence score indicating how likely 
    it is that the investor will invest in that company. These ranked lists are the target lists that we provide 
    to customers, advising them on which investors to approach regarding the target company.

    Evaluation with HR and NDCG:
        HR@k (Hit Ratio): Measures the proportion of investors for whom the target company appears within the 
        top-k predictions of the model. A high HR@k indicates that many investors are correctly predicted to 
        eventually invest in the target company.
        NDCG@k (Normalized Discounted Cumulative Gain): Evaluates not only if the target company appears in the 
        list, but also how high it is ranked. This metric gives a more nuanced view of ranking quality, ensuring 
        that the most promising investor candidates are placed at the top of the target list.

In Short:
    Data Source: Holdings data capturing past investor behavior.
    Processing: You build sequential interaction data per investor and train a model (e.g., SASRec) to predict 
                the next company they might invest in.
    Output: For a given target company, the model produces a ranked list (target list) of investors with 
            confidence scores.
    Customer Value: These target lists help customers (e.g., investment teams) focus outreach on investors 
                    who are most likely to be interested in the target company based on their historical behavior.


NOTE TECHNICAL DISCUSSION:

NOTE: In a standard SASRec setup, the model is trained on sequences derived from investors' past holdings 
(interaction data). Here, the sequential examples consist of each investor's ordered list of investments. 
The orgs data (which may include bios and other metadata) could later be used to enrich the investor representations 
in a hybrid or two-tower model (e.g., A-LLMRec). For SASRec alone, however, the primary input is the sequential data.

MAJOR BUG: 
The timestamp data in our CSV is not reliable (all values are the same), so the current ordering of transactions 
is based solely on the row order in the CSV. Consider providing or engineering proper timestamp data to ensure 
accurate temporal ordering.

Evaluation:
    In the context of this assignment, HR@k and NDCG@k are used to evaluate how well the recommender system 
    generates target lists of investors for a given target company.
        - HR@k: Checks if the target company (i.e., the one the investor eventually invests in) appears within 
          the top-k predicted companies for that investor. When filtering by the target company, it tells you if 
          the investor is a good candidate.
        - NDCG@k: Measures not only the presence but also the ranking of the target company in the list. A higher 
          rank (closer to the top) contributes more to NDCG, indicating that the investor is highly likely to invest 
          in the target company.

When running the model (such as SASRec or a two-tower model like A-LLMRec), these metrics are computed on the test 
or validation set by comparing the predicted “next investment” for each investor to the actual target company. 
This process results in confidence scores for each investor regarding their likelihood to invest in the target company.
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
        dynamic_ncols=True,
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df


def preprocess(df_holdings, processed_org_path, seq_out_file):
    """
    Process the holdings CSV file to generate:
      - Counts of interactions per investor and per company.
      - Mapping dictionaries (usermap and itemmap) to assign new integer IDs.
      - A User dictionary mapping each new investor ID (derived from the original filer_id)
        to a list of interactions, where each interaction is represented as [timestamp, itemid].
        Since no real timestamp is available, we assign one by incrementing a counter.
      - An org_dict dictionary mapping each new investor ID to metadata obtained from the orgs CSV
        (using the 'bio' field and optionally 'stock_ticker').

    Note:
      In our holdings data, "org_id" or "org_id_encoded" does NOT represent the investor;
      the investor is identified by "filer_id". Thus, we use "filer_id" as the key for building
      the investor (user) mappings.

    The function also saves:
      - org_dict as a gzipped pickle file.
      - A text file with one interaction per line in the format "userid itemid".

      User (dict): Mapping from new investor IDs (derived from filer_id) to lists of interactions over "time".
      usermap (dict): Mapping from original filer_id to new integer IDs.
      itemmap (dict): Mapping from original stock_id to new integer IDs.
    """
    from collections import defaultdict
    import pandas as pd
    import gzip, pickle

    # --- Counting interactions from df_holdings ---
    countU = defaultdict(int)  # Investor interaction counts (using filer_id)
    countP = defaultdict(int)  # Company interaction counts (using stock_id)
    for idx, row in df_holdings.iterrows():
        investor = row["filer_id"]
        stock = row["stock_id"]
        countU[investor] += 1
        countP[stock] += 1

    # --- Building mappings and sequential data ---
    usermap = {}  # Maps original filer_id to new integer investor IDs.
    itemmap = {}  # Maps original stock_id to new integer stock IDs.
    User = {}  # Maps new investor IDs to lists of interactions ([timestamp, itemid])
    usernum = 0
    itemnum = 0
    threshold = 4  # Minimum interactions required.
    timestamp_counter = 0  # Dummy timestamp counter (since real timestamps are absent)

    for idx, row in df_holdings.iterrows():
        investor = row["filer_id"]
        stock = row["stock_id"]
        timestamp = timestamp_counter
        timestamp_counter += 1

        # Only include interactions if both investor and stock have enough interactions.
        if countU[investor] < threshold or countP[stock] < threshold:
            continue

        # Map the investor (filer_id) to a new integer ID.
        if investor not in usermap:
            usernum += 1
            usermap[investor] = usernum
            User[usernum] = []
        userid = usermap[investor]

        # Map the stock (stock_id) to a new integer ID.
        if stock not in itemmap:
            itemnum += 1
            itemmap[stock] = itemnum
        itemid = itemmap[stock]

        # Append the interaction as [timestamp, itemid].
        User[userid].append([timestamp, itemid])

    # --- Enriching stock metadata from df_orgs ---
    df_orgs = pd.read_csv(processed_org_path)
    df_stocks = pd.read_csv("guilherme/data/raw/stocks.csv")
    item_meta_dict = {
        "bio": {},
        "stock_ticker": {},
        "stock_industry": {},
        "stock_sector": {},
        "company_name": {},
    }
    missing_count = 0
    for stock_id, new_id in itemmap.items():
        stocks_row = df_stocks[df_stocks["id"] == stock_id]
        stock_ticker = stocks_row["symbol"]
        company_name = stocks_row["name"]
        stock_sector = stocks_row["sector"]
        stock_industry = stocks_row["industry"]
        # may not always have bio
        orgs_row = df_orgs[df_orgs["stock_id"] == stock_id]
        if not orgs_row.empty:
            bio = orgs_row["bio"]
        else:
            missing_count += 1
            bio = "No bio available"
        item_meta_dict[new_id] = {
            "bio": bio,
            "stock_ticker": stock_ticker,
            "stock_industry": stock_industry,
            "stock_sector": stock_sector,
            "company_name": company_name,
        }
    print(f"Missing {missing_count} item (stock) bios!")
    # Save the org_dict as a gzipped pickle file.
    org_dict_path = "guilherme/data/processed/holdings_org_dict.pkl.gz"
    with gzip.open(org_dict_path, "wb") as tf:
        pickle.dump(item_meta_dict, tf)

    print("Total new investor IDs:", usernum, "Total new stock IDs:", itemnum)

    # --- Sorting interactions per investor ---
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    # --- Writing the interactions to a text file ---
    with open(seq_out_file, "w") as f:
        for user in User.keys():
            for interaction in User[user]:
                f.write("%d %d\n" % (user, interaction[1]))


##############################################
# SASRec Main Routine
##############################################

"""
For a baseline like SASRec, the standard setup relies on the sequential patterns from historical interactions 
to learn user (investor) embeddings implicitly.This means that, by default, the model learns user 
representations purely from the sequence of investments.
"""
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
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=0.0008, type=float)
parser.add_argument("--maxlen", default=54, type=int)  # around the avg
parser.add_argument("--hidden_units", default=128, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=300, type=int)
parser.add_argument("--num_heads", default=2, type=int)
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
    "position_change_type",
    "current_ranking",
    "current_percent_of_portfolio",
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
        df = read_csv_with_progress(args.dataset, chunksize=256, total_rows=173944)
        print("Columns in the DataFrame:", df.columns.tolist())
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

    # Hack -- can include OHE features (try low cardinality), too
    if args.holdings_extra_features:
        # Build a lookup dictionary: item_id -> extra features vector; OHE arent usually too helpful with
        # enriching due to sparsity (inherent orthgonality not conducive to sim measure) not leading to meaningful rel.
        # bet categories so opt for sector for a balance?
        extra_one_hot = [
            col for col in df.columns if col.startswith("PROBABLYDONTINCLUDE")
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
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1), dynamic_ncols=True):
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
            batch_extra_feats = (
                torch.from_numpy(np.array(batch_extra_feats)).float().to(args.device)
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

        if epoch % 50 == 0 or epoch == 1:
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
