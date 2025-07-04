import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class GraphAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, keys, values, relevance_mask=None):
        q = self.query_proj(query).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        k = self.key_proj(keys)  # [batch_size, num_graph_nodes, hidden_dim]
        v = self.value_proj(values)  # [batch_size, num_graph_nodes, hidden_dim]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, 1, num_graph_nodes]
        if relevance_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~relevance_mask.unsqueeze(1),  # Positions with True will be filled with -inf
                float('-inf')
            )
        if attention_scores.numel() > 0:
            is_all_neg_inf_row = torch.all(torch.isneginf(attention_scores), dim=2).squeeze(1)  # Result shape [batch_size]
            if is_all_neg_inf_row.any():
                logger.warning(
                    f"GraphAttLayer: Found {is_all_neg_inf_row.sum().item()} samples where all attention scores are -inf before softmax!")

        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, num_graph_nodes]

        # Handle potential NaNs from softmax (when all inputs are -inf)
        if torch.isnan(attention_weights).any():
            logger.error("GraphAttLayer: !!! NaN detected in attention_weights after softmax !!!")
            # Replace NaN with 0, so the context_vector for these samples will be 0
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
            logger.warning("GraphAttLayer: Replaced NaN in attention_weights with 0.0.")

        attention_weights_for_return = attention_weights.squeeze(1)  # [batch_size, num_graph_nodes]
        context_vector = torch.matmul(attention_weights, v).squeeze(1)  # [batch_size, hidden_dim]


        return context_vector, attention_weights_for_return


class EnhancedEconomicStateAwareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes, num_economic_states=5,
                 gsmd_node_embed_size=32, num_rmesr_categories=7, target_names=None,
                 use_context_gate=False, use_graph_attention=True,
                 graph_attention_hidden_dim=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes
        self.num_economic_states = num_economic_states
        self.gsmd_node_embed_size = gsmd_node_embed_size  # Original GSMD node embedding dimension from GAT output
        self.num_rmesr_categories = num_rmesr_categories
        self.target_names = target_names
        self.use_context_gate = use_context_gate
        self.use_graph_attention = use_graph_attention
        self.graph_attention_hidden_dim = graph_attention_hidden_dim

        self.econ_state_embedding_dim = 8
        self.econ_state_embedding = nn.Embedding(num_economic_states, self.econ_state_embedding_dim)

        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.lstm_input_dim_base = hidden_size + self.econ_state_embedding_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim_base,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        if self.use_graph_attention:
            self.graph_attention_module = GraphAttentionLayer(
                query_dim=hidden_size,
                key_dim=gsmd_node_embed_size,
                hidden_dim=self.graph_attention_hidden_dim
            )
            self.fused_lstm_output_dim = hidden_size + self.graph_attention_hidden_dim
            logger.info(
                f"EnhancedEconomicStateAwareLSTM: Using graph attention. Fused LSTM output related dimension: {self.fused_lstm_output_dim}")
        else:
            self.processed_missing_emb_dim_fallback = 8
            self.missing_embed_processor_fallback = nn.Sequential(
                nn.Linear(gsmd_node_embed_size, 16),
                nn.ReLU(),
                nn.Linear(16, self.processed_missing_emb_dim_fallback)
            )
            self.fused_lstm_output_dim = hidden_size
            logger.info("EnhancedEconomicStateAwareLSTM: Not using graph attention. May use fallback for missing embedding processing.")

        if self.use_graph_attention:
            head_input_dim = self.fused_lstm_output_dim + self.econ_state_embedding_dim
            state_predictor_input_dim = self.fused_lstm_output_dim
        else:
            head_input_dim = self.hidden_size + self.processed_missing_emb_dim_fallback + self.econ_state_embedding_dim
            state_predictor_input_dim = self.hidden_size

        self.task_heads = nn.ModuleList()
        rmesr_head_idx = -1
        if self.target_names and self.num_rmesr_categories > 0:
            try:
                for i, name in enumerate(self.target_names):
                    if name == 'RMESR' and i < len(self.output_sizes) and self.output_sizes[
                        i] == self.num_rmesr_categories:
                        rmesr_head_idx = i;
                        break
                if rmesr_head_idx == -1 and 'RMESR' in self.target_names: logger.warning(
                    f"RMESR is in target_names, but output_size/num_rmesr_categories do not match or index is out of bounds.")
            except ValueError:
                pass
        elif self.num_rmesr_categories > 0:  # Try to infer when target_names is not provided
            for i, size_ in enumerate(self.output_sizes):
                if size_ == self.num_rmesr_categories: rmesr_head_idx = i; logger.info(f"Inferred RMESR head index as: {i}"); break
        if rmesr_head_idx != -1: logger.info(
            f"RMESR classification head index: {rmesr_head_idx}, output dimension: {self.output_sizes[rmesr_head_idx]}")

        for i, size in enumerate(self.output_sizes):
            self.task_heads.append(nn.Sequential(
                nn.Linear(head_input_dim, hidden_size // 2), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, size), nn.Identity()
            ))
        self.state_predictor = nn.Sequential(
            nn.Linear(state_predictor_input_dim, hidden_size), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size, num_economic_states)
        )

    def forward(self, x, economic_state=None,
                batch_graph_node_embeddings=None,
                batch_graph_nodes_relevance_mask=None,
                prev_state_lstm=None):

        batch_size, seq_len, _ = x.shape

        mask = torch.isnan(x)
        x_replaced = torch.where(mask, torch.zeros_like(x), x)
        projected_features = self.feature_proj(x_replaced)
        if torch.isnan(projected_features).any(): raise ValueError("NaN in projected_features!")

        if economic_state is None:
            economic_state_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        else:
            economic_state_indices = economic_state.squeeze()
        if economic_state_indices.dim() == 0: economic_state_indices = economic_state_indices.unsqueeze(0)
        if economic_state_indices.dim() == 1:
            econ_emb_seq = self.econ_state_embedding(economic_state_indices).unsqueeze(1).expand(-1, seq_len, -1)
            last_step_econ_emb_for_heads = self.econ_state_embedding(economic_state_indices)
        elif economic_state_indices.dim() == 2:
            econ_emb_seq = self.econ_state_embedding(economic_state_indices)
            last_step_econ_emb_for_heads = econ_emb_seq[:, -1, :]
        else:
            logger.error(f"Unsupported econ_state_indices dim: {economic_state_indices.dim()}. Using zeros.")
            econ_emb_seq = torch.zeros(batch_size, seq_len, self.econ_state_embedding_dim, device=x.device)
            last_step_econ_emb_for_heads = torch.zeros(batch_size, self.econ_state_embedding_dim, device=x.device)

        if torch.isnan(econ_emb_seq).any(): raise ValueError("NaN in econ_emb_seq!")

        lstm_input_seq_base = torch.cat([projected_features, econ_emb_seq], dim=-1)
        current_lstm_module_input_seq = lstm_input_seq_base
        processed_missing_emb_fallback_for_heads = None

        if not self.use_graph_attention:
            if batch_graph_node_embeddings is not None and batch_graph_node_embeddings.numel() > 0:
                if batch_graph_nodes_relevance_mask is not None:
                    masked_embeddings = batch_graph_node_embeddings * batch_graph_nodes_relevance_mask.unsqueeze(
                        -1).float()
                    sum_masked_embeddings = masked_embeddings.sum(dim=1)
                    num_relevant_nodes = batch_graph_nodes_relevance_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
                    avg_relevant_node_emb = sum_masked_embeddings / num_relevant_nodes
                else:
                    avg_relevant_node_emb = batch_graph_node_embeddings.mean(dim=1)
                processed_avg_emb = self.missing_embed_processor_fallback(avg_relevant_node_emb)
                processed_missing_emb_fallback_seq = processed_avg_emb.unsqueeze(1).expand(-1, seq_len, -1)
                current_lstm_module_input_seq = torch.cat(
                    [current_lstm_module_input_seq, processed_missing_emb_fallback_seq], dim=-1)
                processed_missing_emb_fallback_for_heads = processed_avg_emb
            else:
                zeros_fallback_missing = torch.zeros(batch_size, seq_len, self.processed_missing_emb_dim_fallback,
                                                     device=x.device)
                current_lstm_module_input_seq = torch.cat([current_lstm_module_input_seq, zeros_fallback_missing],
                                                          dim=-1)
                processed_missing_emb_fallback_for_heads = torch.zeros(batch_size,
                                                                       self.processed_missing_emb_dim_fallback,
                                                                       device=x.device)

        if torch.isnan(current_lstm_module_input_seq).any(): raise ValueError(
            "NaN in current_lstm_module_input_seq before LSTM module!")

        final_lstm_hidden_state_tuple = None
        lstm_processed_output_seq = None

        if self.use_graph_attention:
            if batch_graph_node_embeddings is None or batch_graph_node_embeddings.numel() == 0:
                logger.warning(
                    "use_graph_attention=True, but batch_graph_node_embeddings is empty. Graph attention may be ineffective.")
                num_dummy_nodes = 1
                model_gsmd_embed_size = getattr(self, 'gsmd_node_embed_size', 32)  # Get from self
                batch_graph_node_embeddings = torch.zeros(batch_size, num_dummy_nodes, model_gsmd_embed_size,
                                                          device=x.device)
                if batch_graph_nodes_relevance_mask is None:
                    batch_graph_nodes_relevance_mask = torch.ones(batch_size, num_dummy_nodes, dtype=torch.bool,
                                                                  device=x.device)

            lstm_outputs_fused_list = []
            lstm_state_for_loop = prev_state_lstm

            for t in range(seq_len):
                lstm_module_input_t = current_lstm_module_input_seq[:, t, :].unsqueeze(1)

                output_t_raw, next_lstm_state_tuple = self.lstm(lstm_module_input_t, lstm_state_for_loop)

                if torch.isnan(output_t_raw).any(): raise ValueError(f"NaN in LSTM output_t_raw at t={t}!")

                query_for_graph_att = next_lstm_state_tuple[0][-1]  # last layer of hn_t
                if torch.isnan(query_for_graph_att).any(): raise ValueError(f"NaN in query_for_graph_att at t={t}!")

                graph_context_t = torch.zeros(batch_size, self.graph_attention_hidden_dim,
                                              device=query_for_graph_att.device)
                attention_weights_t_for_log = torch.zeros(batch_size, batch_graph_node_embeddings.shape[1],
                                                          device=query_for_graph_att.device)

                sample_has_relevant_nodes_mask = batch_graph_nodes_relevance_mask.any(dim=1)

                if sample_has_relevant_nodes_mask.any():
                    active_sample_indices = sample_has_relevant_nodes_mask.nonzero(as_tuple=False).squeeze(-1)

                    sub_query = query_for_graph_att[active_sample_indices]
                    sub_keys = batch_graph_node_embeddings[active_sample_indices]
                    sub_values = batch_graph_node_embeddings[active_sample_indices]
                    sub_relevance_mask_for_attn = batch_graph_nodes_relevance_mask[active_sample_indices]

                    # Reconfirm if there are any True values in the sub-mask, just in case
                    if sub_relevance_mask_for_attn.any():
                        computed_graph_context, computed_attention_weights = self.graph_attention_module(
                            sub_query, sub_keys, sub_values, sub_relevance_mask_for_attn
                        )
                        # Use scatter_ to update, ensuring dimension matching
                        graph_context_t.scatter_(0,
                                                 active_sample_indices.unsqueeze(1).expand_as(computed_graph_context),
                                                 computed_graph_context)
                        attention_weights_t_for_log.scatter_(0, active_sample_indices.unsqueeze(1).expand_as(
                            computed_attention_weights), computed_attention_weights)
                    else:
                        logger.warning(
                            f"    t={t}: No relevant nodes in 'active_indices' sub-batch (sub_relevance_mask_for_attn is all False). Graph context will be zero.")
                if torch.isnan(graph_context_t).any(): raise ValueError(
                    f"NaN in graph_context_t at t={t}!")

                fused_output_t = torch.cat([output_t_raw.squeeze(1), graph_context_t], dim=-1)
                lstm_outputs_fused_list.append(fused_output_t)
                lstm_state_for_loop = next_lstm_state_tuple

            lstm_processed_output_seq = torch.stack(lstm_outputs_fused_list, dim=1)
            last_step_processed_output_for_heads = lstm_processed_output_seq[:, -1, :]
            final_lstm_hidden_state_tuple = lstm_state_for_loop
        else:
            lstm_processed_output_seq, final_lstm_hidden_state_tuple = self.lstm(current_lstm_module_input_seq,
                                                                                 prev_state_lstm)
            last_step_processed_output_for_heads = lstm_processed_output_seq[:, -1, :]
            #
            if torch.isnan(lstm_processed_output_seq).any(): raise ValueError(
                "NaN in lstm_processed_output_seq (non-graph-att)!")

        input_for_state_predictor = last_step_processed_output_for_heads[:, :self.hidden_size]
        if self.use_graph_attention:
            input_for_state_predictor = last_step_processed_output_for_heads
        predicted_state_logits = self.state_predictor(input_for_state_predictor)

        if torch.isnan(predicted_state_logits).any(): raise ValueError("NaN in predicted_state_logits!")

        base_for_heads = [last_step_processed_output_for_heads, last_step_econ_emb_for_heads]
        if not self.use_graph_attention and processed_missing_emb_fallback_for_heads is not None:
            base_for_heads.insert(1, processed_missing_emb_fallback_for_heads)
        combined_output_for_heads = torch.cat(base_for_heads, dim=-1)

        if torch.isnan(combined_output_for_heads).any(): raise ValueError("NaN in combined_output_for_heads!")

        task_predictions = []
        for i_head, head in enumerate(self.task_heads):
            pred_this_head = head(combined_output_for_heads)

            if torch.isnan(pred_this_head).any(): raise ValueError(f"NaN in prediction from task_head {i_head}!")
            task_predictions.append(pred_this_head)

        all_predictions = torch.cat(task_predictions, dim=1)

        if torch.isnan(all_predictions).any(): raise ValueError("NaN in final all_predictions!")

        return all_predictions, predicted_state_logits, lstm_processed_output_seq, final_lstm_hidden_state_tuple