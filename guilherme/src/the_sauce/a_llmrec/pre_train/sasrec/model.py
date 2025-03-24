import numpy as np
import torch
import torch.nn as nn

"""
In our context, collaborative filtering refers to a recommendation method that 
learns patterns solely from historical interaction data—in this case,
the investment entities' holdings (from filings or transaction records). 
Here's what that means for our us:

    Investor-Stock Interactions:
    We treat each investment entity (e.g., an institutional investor) as a “user” and each 
    stock as an “item.” The historical record of which stocks each investor 
    holds is used as the interaction data.

    Learning Latent Representations:
    A collaborative filtering model (the RecSys component) learns latent embeddings 
    for both investors and stocks based on these interactions. 
    These embeddings capture hidden factors such as investment strategies or 
    stock characteristics that are common among similar investors or stocks.

    Prediction by Similarity:
    Once the model has learned these representations, it can predict 
    the likelihood that an investor would be interested in a particular 
    stock—essentially recommending stocks that “fit” well with the investor's past behavior.

    Why It Matters for Us:
    In our assignment, while we also incorporate content features (like stock bios and tickers) 
    through text embeddings, the collaborative filtering part gives us a signal based 
    solely on historical investment patterns. This helps the system suggest stocks that
    are popular or relevant among similar investors, even before considering additional textual information.
    
Thus, we  use observed interactions (i.e., which stocks are held by which investors) to learn 
an underlying structure of investor preferences and stock characteristics, enabling the recsys 
to make personalized suggestions.



"""


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args, combined_feat_dim=8):
        super(SASRec, self).__init__()

        self.kwargs = {"user_num": user_num, "item_num": item_num, "args": args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.args = args

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # New: Projection layer for combined extra item features.
        # combined_feat_dim is the sum of the dimensions of your one-hot vectors and additional numeric features.
        self.extra_feat_proj = nn.Linear(combined_feat_dim, args.hidden_units)
        nn.init.xavier_normal_(self.extra_feat_proj.weight)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        positions = (
            torch.arange(seqs.size(1), device=self.dev)
            .unsqueeze(0)
            .expand(seqs.size(0), -1)
        )
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.size(1)
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(
        self,
        user_ids,
        log_seqs,
        pos_seqs,
        neg_seqs,
        item_side_features=None,
        mode="default",
    ):
        log_feats = self.log2feats(log_seqs)
        if mode == "log_only":
            log_feats = log_feats[:, -1, :]
            return log_feats

        # Get base item embeddings.
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # Enrich item embeddings if extra features are provided.
        if item_side_features is not None:
            # item_side_features: tensor of shape [batch_size, seq_len, combined_feat_dim]
            extra_proj = self.extra_feat_proj(
                item_side_features
            )  # shape: [batch_size, seq_len, hidden_units]
            # Fuse by addition:
            pos_embs = pos_embs + extra_proj
            neg_embs = neg_embs + extra_proj

        if mode == "default":
            # Sums over the last dimension (which effectively computes a dot product); can then be used for computing
            # a loss (e.g., BCE loss) during training
            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
            return pos_logits, neg_logits
        elif mode == "item":
            """
            This is useful if you need to operate on or analyze the individual embeddings
            (e.g., for re-ranking or for generating predictions at each timestep)
            rather than computing a dot product for a loss.  reshapes the tensors log_feats,
            pos_embs, and neg_embs so that the batch and sequence dimensions are flattened into one.
            This converts each from a 3D tensor (e.g., [batch_size, seq_len, hidden_units])
            into a 2D tensor (e.g., [batch_size * seq_len, hidden_units]).
            """
            return (
                log_feats.reshape(-1, log_feats.shape[2]),
                pos_embs.reshape(-1, log_feats.shape[2]),
                neg_embs.reshape(-1, log_feats.shape[2]),
            )
        else:
            return

    def predict(self, user_ids, log_seqs, item_indices):
        """
        Only the final hidden state (which represents the user's most recent positive interaction)
        is used to compare against candidate item embeddings. This is because that final state
        is intended to capture the users current interest, which is then used to rank candidate items.
        The idea is that the user's current state (derived from their positive interactions)
        should be used to generate a recommendation list. The negative examples
        are only used during training to help the model learn what not to recommend.
        """
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
