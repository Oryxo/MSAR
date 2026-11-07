import torch
from torch import nn
# import numpy as np
from einops import rearrange
from layers.Masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
from layers.linear_extractor_cluster import Linear_extractor_cluster

class Model(nn.Module):
    def __init__(self, config, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = config.task_name
        
        self.cluster = Linear_extractor_cluster(config)
        
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
                
        self.args = config

        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=0,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation="gelu",
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.linear_head = nn.Sequential(nn.Linear(config.d_model, self.pred_len), nn.Dropout(config.fc_dropout))
        elif self.task_name == 'imputation':
            self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.seq_len), nn.Dropout(config.fc_dropout))

    def forecast(self, input, x_mark_enc, x_dec, x_mark_dec):
        
        means = input.mean(1, keepdim=True).detach()
        input = input - means
        stdev = torch.sqrt(
            torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input = input.div(stdev)

        # x: [batch_size, seq_len, n_vars]

        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature, L_importance = self.cluster(input)
            

        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')

        output = temporal_feature
        output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        
        output = output * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return output, L_importance

    def imputation(self, input, x_mark_enc, x_dec, x_mark_dec, mask):
        
        means = torch.sum(input, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        input = input - means
        input = input.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(input * input, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        input = input.div(stdev)

        # x: [batch_size, seq_len, n_vars]

        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature, L_importance = self.cluster(input)
            

        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')

        output = temporal_feature
        output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        
        output = output * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        output = output + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return output, L_importance

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, joint=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, L_importance = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], L_importance
        if self.task_name == 'imputation':
            dec_out, L_importance = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out[:, -self.seq_len:, :], L_importance