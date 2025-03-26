import random
import pickle
import math
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer
import gzip

"""
NOTE Losses and Their Roles

Our overall goal is to build a recommender system that learns to rank the 
stocks (and their textual content such as bios and tickers) in a way that reflects
the investor's true interests. To achieve this, we use several losses that work together:

    Binary Cross-Entropy (BCE) Loss for Ranking (BPR Loss):

        What It Does:
        This loss compares the predicted scores for positive and negative instances.

        Why It's Used:
        It forces the model to assign higher scores to the positive (held) 
        stocks and lower scores to the negatives. Essentially, it trains the model to 
        correctly distinguish between stocks an investor actually holds versus those they do not.

    Matching Loss (MSE Loss):

        What It Does:
        After projecting the collaborative filtering (interaction-based) embeddings 
        and the text-based embeddings (derived from stock bios and tickers via SBERT) through separate MLPs, 
        we compute the mean squared error (MSE) between these projections.

        Why It's Used:
        This loss encourages the two types of embeddings to align. 
        The idea is that the latent representation learned from the investment 
        interactions should be consistent with the content-based representation 
        derived from stock bios/tickers. When these match well, the model can 
        better capture the underlying semantics of the investor's interests.

    Reconstruction Losses (MSE Loss):

        Item Reconstruction Loss:
        Measures the difference between the original collaborative filtering embeddings
        and their MLP-transformed versions.

        Text Reconstruction Loss:
        Measures the difference between the original SBERT text 
        embeddings and their projected representations.

        Why They're Used:
        These losses ensure that the transformations applied by 
        the MLPs do not distort the original information too much. 
        They help in preserving the meaningful structure of the original 
        embeddings while still allowing the model to align and fuse 
        the collaborative and content information.
"""


class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1


class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = "guilherme/data/processed/holdings_org_dict.pkl.gz"
        self.args = args
        self.device = args.device

        with gzip.open(rec_pre_trained_data, "rb") as ft:
            self.text_name_dict = pickle.load(ft)

        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768

        self.mlp = two_layer_mlp(self.rec_sys_dim)
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer("nq-distilbert-base-v1")
            self.mlp2 = two_layer_mlp(self.sbert_dim)

        self.mse = nn.MSELoss()

        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)

            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(
                    self.llm.llm_model.config.hidden_size,
                    self.llm.llm_model.config.hidden_size,
                ),
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(
                    self.llm.llm_model.config.hidden_size,
                    self.llm.llm_model.config.hidden_size,
                ),
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)

    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = "guilherme/src/the_sauce/saved_models/"
        create_dir(out_dir)
        out_dir += "a_llmrec" + f"_{args.recsys}_{epoch1}_"
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + "sbert.pt")
            torch.save(self.mlp.state_dict(), out_dir + "mlp.pt")
            torch.save(self.mlp2.state_dict(), out_dir + "mlp2.pt")

        out_dir += f"{args.llm}_{epoch2}_"
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + "log_proj.pt")
            torch.save(self.item_emb_proj.state_dict(), out_dir + "item_proj.pt")

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = "guilherme/src/the_sauce/saved_models/"
        out_dir += "a_llmrec" + f"_{args.recsys}_{phase1_epoch}_"

        mlp = torch.load(out_dir + "mlp.pt", map_location=args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            out_dir += f"{args.llm}_{phase2_epoch}_"

            log_emb_proj_dict = torch.load(
                out_dir + "log_proj.pt", map_location=args.device
            )
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict

            item_emb_proj_dict = torch.load(
                out_dir + "item_proj.pt", map_location=args.device
            )
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = "stock_ticker"
        d = "bio"
        t_ = "Unknown"
        d_ = "No bio available"
        if title_flag and description_flag:
            return [
                f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"'
                for i in item
            ]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = "stock_ticker"
        d = "bio"
        t_ = "Unknown"
        d_ = "No bio available"
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'

    def get_item_emb(self, item_ids):
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(
                torch.LongTensor(item_ids).to(self.device)
            )
            item_embs, _ = self.mlp(item_embs)

        return item_embs

    def forward(self, data, args, optimizer=None, batch_iter=None, mode="phase1"):
        if mode == "phase1":
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == "phase2":
            self.pre_train_phase2(data, args, optimizer, batch_iter)
        # if mode == "generate":
        #    self.generate(data)
        if mode == "generate_target_lists_for_companies":
            self.generate_target_lists_for_companies(args)

    def pre_train_phase1(self, data, optimizer, batch_iter):
        """
        Pre-train Phase 1 for the A-LLMRec model.

        In our recommendation problem, this phase trains the collaborative filtering
        and text alignment components. We first obtain item embeddings from our recsys
        model, then iterate in batches to:
        - Generate text representations of positive and negative items via SBERT.
        - Project the recsys embeddings and text embeddings through separate MLPs.
        - Compute a binary cross-entropy (BCE) loss for ranking (BPR loss),
            an MSE-based matching loss to align the embeddings,
            and reconstruction losses.
        - Backpropagate and update the model.

        Positive Instances:
        These are the items (stocks) that an investment entity actually
        holds or has interacted with. In our data (e.g., from the 13F filings),
        a positive instance is a stock that is present in an investor's portfolio.
        For example, if an institutional investor holds shares of Company X, then Company X
        is a positive instance for that investor.

        Negative Instances:
        These are items that the investment entity does not hold.
        In training the model, negative instances are usually sampled at random
        from the set of all stocks—ensuring that they are not in the investor's historical holdings.
        The model learns to assign lower relevance scores to these
        negative instances compared to the positive ones.

        Parameters:
            data: a tuple (u, seq, pos, neg) where:
                - u: user representation tensor.
                - seq: sequences (historical interactions).
                - pos: positive item interactions.
                - neg: negative item interactions.
            optimizer: The optimizer used to update model parameters.
            batch_iter: A tuple (epoch, total_epoch, step, total_step) providing context for logging.
        """
        # Unpack batch_iter context
        epoch, total_epoch, step, total_step = batch_iter

        # Set the SBERT model to training mode
        self.sbert.train()
        optimizer.zero_grad()

        # Unpack input data: u, sequence, positive and negative items.
        u, seq, pos, neg = data
        # Compute indices: for each user, pick the index corresponding to the last item in their sequence.
        indices = [self.maxlen * (i + 1) - 1 for i in range(u.shape[0])]

        # Get initial item embeddings (log_emb) from the recsys model;
        # these embeddings come solely from interaction data.
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode="item")

        # Select embeddings corresponding to the last item in each sequence.
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        # Reshape positive and negative item tensors accordingly.
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]

        # BUG terrible. should not be hardcoded. Come from args.
        batch_size = 64
        num_iters = math.ceil(len(log_emb_) / batch_size)

        # Initialize loss accumulators.
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0

        # Use tqdm to wrap the iteration for a nice progress bar.
        for i in tqdm(
            range(num_iters), desc="Phase1 iterations", leave=False, dynamic_ncols=True
        ):
            start_inx = i * batch_size
            end_inx = min(start_inx + batch_size, len(log_emb_))

            # Extract the current mini-batch.
            log_emb_batch = log_emb_[start_inx:end_inx]
            pos_emb_batch = pos_emb_[start_inx:end_inx]
            neg_emb_batch = neg_emb_[start_inx:end_inx]
            pos_batch = pos_[start_inx:end_inx]
            neg_batch = neg_[start_inx:end_inx]

            # Obtain text representations of the positive and negative items.
            # These functions retrieve the item title (stock id for ticker) and bio to form a text description.
            pos_text = self.find_item_text(pos_batch)
            neg_text = self.find_item_text(neg_batch)

            # Tokenize and embed positive text using SBERT.
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding = self.sbert(
                {
                    "input_ids": pos_token["input_ids"].to(self.device),
                    "attention_mask": pos_token["attention_mask"].to(self.device),
                }
            )["sentence_embedding"]

            # Tokenize and embed negative text using SBERT.
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding = self.sbert(
                {
                    "input_ids": neg_token["input_ids"].to(self.device),
                    "attention_mask": neg_token["attention_mask"].to(self.device),
                }
            )["sentence_embedding"]

            # Project recsys embeddings through MLP to obtain a new representation.
            pos_text_matching, pos_proj = self.mlp(pos_emb_batch)
            neg_text_matching, neg_proj = self.mlp(neg_emb_batch)
            # Similarly, project text embeddings via a second MLP.
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)

            # Compute dot-product based logits between the log_emb_batch and projected embeddings.
            pos_logits = (log_emb_batch * pos_proj).mean(axis=1)
            neg_logits = (log_emb_batch * neg_proj).mean(axis=1)
            # Create target labels for positive (1) and negative (0) interactions.
            pos_labels = torch.ones(pos_logits.shape, device=pos_logits.device)
            neg_labels = torch.zeros(neg_logits.shape, device=pos_logits.device)

            # Calculate the binary cross-entropy loss for ranking (BPR loss).
            loss = self.bce_criterion(pos_logits, pos_labels) + self.bce_criterion(
                neg_logits, neg_labels
            )
            # Matching loss aligns the recsys MLP output with text MLP output.
            matching_loss = self.mse(
                pos_text_matching, pos_text_matching_text
            ) + self.mse(neg_text_matching, neg_text_matching_text)
            # Reconstruction losses encourage the MLP outputs to remain close to the original embeddings.
            reconstruction_loss = self.mse(pos_proj, pos_emb_batch) + self.mse(
                neg_proj, neg_emb_batch
            )
            text_reconstruction_loss = self.mse(
                pos_text_proj, pos_text_embedding.data
            ) + self.mse(neg_text_proj, neg_text_embedding.data)

            # Combine the losses with specific weighting.
            total_loss = (
                loss
                + matching_loss
                + 0.5 * reconstruction_loss
                + 0.2 * text_reconstruction_loss
            )
            total_loss.backward()
            optimizer.step()

            # Accumulate losses for reporting.
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()

        # Compute average losses across iterations.
        avg_mean_loss = mean_loss / num_iters
        avg_bpr_loss = bpr_loss / num_iters
        avg_gt_loss = gt_loss / num_iters
        avg_rc_loss = rc_loss / num_iters
        avg_text_rc_loss = text_rc_loss / num_iters

        # Use tqdm.write to print out the loss summary (this integrates nicely with the progress bar).
        tqdm.write(
            "Epoch {}/{} Iteration {}/{}: Mean loss: {:.5f} / BPR loss: {:.5f} / Matching loss: {:.5f} / "
            "Item reconstruction: {:.5f} / Text reconstruction: {:.5f}".format(
                epoch,
                total_epoch - 1,
                step,
                total_step,
                avg_mean_loss,
                avg_bpr_loss,
                avg_gt_loss,
                avg_rc_loss,
                avg_text_rc_loss,
            ),
        )

    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_item_text(
            interact_ids, title_flag=True, description_flag=False
        )
        interact_text = []
        if interact_max_num == "all":
            for title in interact_item_titles_:
                interact_text.append(title + "[HistoryEmb]")
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + "[HistoryEmb]")
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ",".join(interact_text)
        return interact_text, interact_ids

    def make_candidate_text(
        self, interact_ids, candidate_num, target_item_id, target_item_title
    ):
        neg_item_id = []
        while len(neg_item_id) < 50:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]
        candidate_text = [target_item_title + "[CandidateEmb]"]

        for neg_candidate in neg_item_id[: candidate_num - 1]:
            candidate_text.append(
                self.find_item_text_single(
                    neg_candidate, title_flag=True, description_flag=False
                )
                + "[CandidateEmb]"
            )
            candidate_ids.append(neg_candidate)

        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]

        return ",".join(candidate_text), candidate_ids

    def pre_train_phase2(self, data, args, optimizer, batch_iter):
        """
        Pre-train Phase 2 for the A-LLMRec model in our stock recommendation problem.

        In this phase, we align collaborative filtering embeddings (derived from investor interactions
        with stocks) with text-based representations of stocks (using their bios, tickers, etc.).

        For each investor (user) in the batch:
        - We retrieve the last positive stock interaction as the target.
        - We generate a textual summary of the investor's historical stock interactions.
        - We construct a candidate set of stocks (using negative sampling) and create a text prompt
            that instructs the model to recommend one next stock.
        - We obtain text embeddings via SBERT and project both the collaborative filtering and text
            embeddings through MLPs.
        - A matching loss is computed to align the two representations.

        The loss is then backpropagated and the optimizer updates the model parameters.

        Parameters:
            data: A tuple (u, seq, pos, neg) where:
                - u: Tensor of user representations.
                - seq: Historical interaction sequences.
                - pos: Positive (held) stock interactions.
                - neg: Negative (not-held) stock interactions.
            optimizer: Optimizer to update model parameters.
            batch_iter: A tuple (epoch, total_epoch, step, total_step) providing logging context.
        """
        epoch, total_epoch, step, total_step = batch_iter

        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0

        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []

        # Set the LLM module to evaluation mode (no dropout, etc.).
        self.llm.eval()

        # Obtain the collaborative filtering (CF) embeddings from the recsys model.
        # Here, "log_only" mode returns user log embeddings that serve as the base for alignment.
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode="log_only")

        # We'll process each user (each row in u) one by one.
        num_users = len(u)
        for i in tqdm(
            range(num_users), desc="Phase2 iterations", leave=False, dynamic_ncols=True
        ):
            # For each investor, use the last positive stock interaction as the target.
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(
                target_item_id, title_flag=True, description_flag=False
            )

            # Generate a textual summary of the investor's historical stock interactions.
            interact_text, interact_ids = self.make_interact_text(
                seq[i][seq[i] > 0], 10
            )
            candidate_num = 20
            # Generate candidate stock set and corresponding text prompt.
            candidate_text, candidate_ids = self.make_candidate_text(
                seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title
            )

            # Get investor (user) info via the updated get_user_info function.
            user_info = self.get_user_info(u[i])

            # Build the input prompt for the LLM.
            input_text = user_info + " is an investor representation. "
            input_text += (
                "This investor has held the following stocks: " + interact_text
            )
            input_text += " in the past. Recommend a list of 10 next stocks for this investor from the following candidate set: "
            input_text += candidate_text + ". The recommended list is "

            text_input.append(input_text)
            text_output.append(target_item_title)

            # Retrieve and project item embeddings for historical stocks and candidate stocks.
            interact_emb = self.get_item_emb(interact_ids)
            candidate_emb = self.get_item_emb(candidate_ids)
            interact_embs.append(self.item_emb_proj(interact_emb))
            candidate_embs.append(self.item_emb_proj(candidate_emb))

        # Package text inputs and embeddings into a dictionary for the LLM.
        samples = {
            "text_input": text_input,
            "text_output": text_output,
            "interact": interact_embs,
            "candidate": candidate_embs,
        }

        # Project the CF embeddings using the log embedding projection.
        log_emb_proj = self.log_emb_proj(log_emb)
        # Compute the loss from the LLM module which aligns the CF and text representations.
        loss_rm = self.llm(log_emb_proj, samples)
        loss_rm.backward()
        optimizer.step()

        mean_loss += loss_rm.item()

        tqdm.write(
            "A-LLMRec model loss in epoch {}/{} iteration {}/{}: {:.5f}".format(
                epoch, total_epoch - 1, step, total_step, mean_loss
            )
        )

    def get_user_info(self, user_tensor):
        """
        Return a textual summary for the investor based on user metadata.
        Loads the metadata from the gzipped pickle file if it hasn't been loaded already.
        """
        # Load user metadata if not already loaded.
        if not hasattr(self, "user_meta_dict"):
            with gzip.open(
                "guilherme/data/processed/filers_org_dict.pkl.gz", "rb"
            ) as tf:
                self.user_meta_dict = pickle.load(tf)

        # Convert the user_tensor (assumed to be an integer or a tensor with one value)
        new_id = int(user_tensor)

        # Retrieve investor metadata from the loaded user_meta_dict.
        parts = []
        if new_id in self.user_meta_dict:
            investor_name = self.user_meta_dict[new_id].get(
                "investor_name", "Unknown investor"
            )
            # Check if investor_name is a NumPy array and extract the first element if necessary.
            if isinstance(investor_name, np.ndarray):
                investor_name = (
                    investor_name.item()
                    if investor_name.size == 1
                    else investor_name[0]
                )
            elif isinstance(investor_name, torch.Tensor):
                investor_name = (
                    investor_name.item()
                    if investor_name.dim() == 0
                    else investor_name[0].item()
                )

            parts.append(investor_name)
            filer_id = self.user_meta_dict[new_id].get("filer_id", "Unknown filer")
            # Check if filer_id is a NumPy array and extract the first element if necessary.
            if isinstance(filer_id, np.ndarray):
                filer_id = filer_id.item() if filer_id.size == 1 else filer_id[0]
            elif isinstance(filer_id, torch.Tensor):
                filer_id = (
                    filer_id.item() if filer_id.dim() == 0 else filer_id[0].item()
                )

            parts.append(str(int(filer_id)))
            return f'"Investor {investor_name} (filer id: {str(int(filer_id))})"'
        else:
            return f"Investor (DATA MISSING POST-PROCESSING)"

    def generate_target_lists_for_companies(self, args, top_k=10):
        """
        Generate target investor lists for companies using 13D data. This version loads investor
        interaction sequences from a preprocessed file and uses them to compute investor embeddings.

        Args:
            args: Should include 'email_extension' (list of decoded email extensions) and 'seq_file' path.
            top_k: Number of top investors to return for each company.

        Returns:
            dict: Mapping each company (by descriptive text) to a list of candidate investors with confidence scores.
        """
        outputs = {}
        import pandas as pd

        # Only consider companies provided as input.
        company_list = args.email_extension

        # 1. Load the email extension map.
        with open("guilherme/data/raw/email_extension_map.pkl", "rb") as f:
            email_extension_map = pickle.load(f)

        # 2. Load processed organization data.
        df_orgs = pd.read_csv("guilherme/data/processed/orgs_processed.csv")

        # 3. Load candidate investor IDs from user_meta_dict.
        if not hasattr(self, "user_meta_dict"):
            with gzip.open(
                "guilherme/data/processed/filers_org_dict.pkl.gz", "rb"
            ) as tf:
                self.user_meta_dict = pickle.load(tf)
        candidate_investor_ids = list(
            {
                item["filer_id"]
                for item in self.user_meta_dict.values()
                if "filer_id" in item
            }
        )

        # Process each target company (only the ones provided in company_list).
        for company in company_list:
            # Convert decoded email to encoded.
            encoded_email = email_extension_map.get(company)
            if encoded_email is None:
                print(
                    f"Warning: '{company}' not found in email_extension_map. Skipping."
                )
                continue

            # Filter orgs data for the target company.
            target_rows = df_orgs[df_orgs["email_extension_encoded"] == encoded_email]
            if target_rows.empty:
                print(
                    f"Warning: No company found for encoded email '{encoded_email}'. Skipping {company}."
                )
                continue

            # Use the first matching row as target info.
            target_info = target_rows.iloc[0].to_dict()
            if "stock_id" not in target_info:
                print(
                    f"Warning: Target company {company} is missing 'stock_id'. Skipping."
                )
                continue

            # 4. Compute the target company's CF embedding.
            company_id = target_info["stock_id"]
            company_emb = self.get_item_emb([company_id])
            company_repr = self.log_emb_proj(company_emb).squeeze(
                0
            )  # shape: (proj_dim,)

            # Get company representation.
            company_emb = self.get_item_emb([company_id])
            company_repr = self.log_emb_proj(company_emb)

            # Compute candidate investor embeddings using the loaded investor_log_seqs.
            investor_log_seqs = load_investor_log_seqs(
                "guilherme/data/processed/sequences.txt", self.maxlen
            )
            investor_embs = self.get_investor_emb(
                candidate_investor_ids, investor_log_seqs
            )
            investor_embs_proj = self.item_emb_proj(investor_embs)
            # 5. Compute candidate investor embeddings.
            # get_investor_emb can accept a list of investor IDs along with investor_log_seqs.
            # (Alternatively, you can loop over candidate_investor_ids and build a tensor)
            investor_embs = self.get_investor_emb(
                candidate_investor_ids, investor_log_seqs
            )  # shape: (num_candidates, proj_dim)
            investor_embs_proj = self.item_emb_proj(
                investor_embs
            )  # Ensure same space as company_repr

            # 6. Compute similarity scores between the target and each candidate investor.
            # Use torch.matmul: result shape is (1, num_candidates); then squeeze to (num_candidates,)
            scores = torch.matmul(
                company_repr.unsqueeze(0), investor_embs_proj.t()
            ).squeeze(0)
            confidence_scores = torch.softmax(scores, dim=0) * 100

            # 7. Select top_k investors.
            top_scores, top_indices = torch.topk(
                confidence_scores, k=min(top_k, confidence_scores.size(0))
            )
            candidate_list = []
            for idx, score in zip(top_indices, top_scores):
                investor_id = candidate_investor_ids[idx]
                investor_name = self.get_user_info(investor_id)
                rationale = f"{investor_name} shows strong affinity based on recent 13D filings."  # You can refine this rationale.
                candidate_list.append(
                    {
                        "investor": investor_name,
                        "confidence": round(float(score.item()), 2),
                        "rationale": rationale,
                    }
                )

            # 8. Get a descriptive text for the target company -- ticker?
            company_text =  target_info["stock_ticker"]
            outputs[company_text] = candidate_list

            # 9. Write results out to file.
            import json

            with open("./investor_target_lists.txt", "a") as f:
                f.write(f"Company: {company_text}\n")
                f.write("Target Investors:\n")
                f.write(json.dumps(candidate_list, indent=4))
                f.write("\n" + "=" * 80 + "\n\n")

        # 10. Return the output dictionary.
        return outputs

    def get_investor_emb(self, investor_ids, investor_log_seqs):
        """
        Given a list of investor IDs and an external mapping (investor_log_seqs) of investor
        interaction sequences, compute each investor's embedding by using the recsys model's
        log2feats method on their historical interaction sequence.

        Args:
            investor_ids (list): List of investor IDs (e.g., new integer IDs).
            investor_log_seqs (dict): Mapping from investor ID to a list of item IDs.

        Returns:
            torch.Tensor: Investor embeddings with shape [N, hidden_units].
        """
        investor_embs = []
        for inv_id in investor_ids:
            # Retrieve the historical sequence for this investor.
            log_seq = investor_log_seqs.get(inv_id)
            if log_seq is None:
                # If no history exists, assign a zero vector.
                emb = torch.zeros(self.recsys.hidden_units).to(self.device)
            else:
                # Convert the sequence to a tensor of shape [1, maxlen]

                log_seq_tensor = torch.LongTensor([log_seq]).to("cpu")
                # Compute the hidden states for the sequence.
                log_feats = self.recsys.model.log2feats(log_seq_tensor)
                # Use the final timestep as the investor's embedding.
                emb = log_feats[:, -1, :].squeeze(0)
            investor_embs.append(emb)
        return torch.stack(investor_embs, dim=0)

def load_investor_log_seqs(seq_file, maxlen):
    """
    Load investor interaction sequences from a text file and return a dictionary mapping each investor
    (user id) to their list of item ids, truncated to maxlen if the sequence exceeds that length.

    Each line in the file is expected to be "userid itemid".

    Args:
        seq_file (str): Path to the sequences file.
        maxlen (int): Maximum sequence length. Sequences longer than maxlen are truncated to the most recent maxlen interactions.
                      Sequences shorter than maxlen remain unchanged.

    Returns:
        dict: Mapping from investor (user) id to a list of item ids (truncated to maxlen if necessary).
    """
    investor_log_seqs = {}
    with open(seq_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            user_id, item_id = int(parts[0]), int(parts[1])
            if user_id not in investor_log_seqs:
                investor_log_seqs[user_id] = []
            investor_log_seqs[user_id].append(item_id)

    # Truncate each sequence to the specified maxlen if necessary.
    for user_id, seq in investor_log_seqs.items():
        if len(seq) > maxlen:
            seq = seq[-maxlen:]  # take the most recent maxlen interactions
        investor_log_seqs[user_id] = seq

    return investor_log_seqs
