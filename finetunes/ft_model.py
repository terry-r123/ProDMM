import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from prodmm.encoder.modeling_prolm import ProLMModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


class MeanPooling(nn.Module):
    """Mean Pooling for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if input_mask is not None:
            # Applying input_mask to zero out masked values
            masked_features = features * input_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / input_mask.sum(dim=1, keepdim=True)
        else:
            mean_pooled_features = torch.mean(features, dim=1)
        return mean_pooled_features


class MeanPoolingProjection(nn.Module):
    """Mean Pooling with a projection layer for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, mean_pooled_features):
        x = self.dropout(mean_pooled_features)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MeanPoolingHead(nn.Module):
    """Mean Pooling Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.mean_pooling_projection = MeanPoolingProjection(config)

    def forward(self, features, input_mask=None):
        mean_pooling_features = self.mean_pooling(features, input_mask=input_mask)
        x = self.mean_pooling_projection(mean_pooling_features)
        return x


class AttentionPoolingHead(nn.Module):
    """Attention Pooling Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.scores = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softmax(dim=1))
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.config = config

    def forward(self, features, input_mask=None):
        attention_scores = self.scores(features).transpose(1, 2)  # [B, 1, L]
        if input_mask is not None:
            # Applying input_mask to attention_scores
            attention_scores = attention_scores * input_mask.unsqueeze(1)
        context = torch.bmm(
            attention_scores, features
        ).squeeze()  # [B, 1, L] * [B, L, D] -> [B, 1, D]
        x = self.dense(context)
        return x


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1dPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = MaskedConv1d(config.hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)
        out = (attn * x).sum(dim=1)
        return out


class Attention1dPoolingProjection(nn.Module):
    def __init__(self, config) -> None:
        super(Attention1dPoolingProjection, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.final = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x


class Attention1dPoolingHead(nn.Module):
    """Outputs of the model with the attention1d"""

    def __init__(
        self, config
    ):  # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(Attention1dPoolingHead, self).__init__()
        self.attention1d = Attention1dPooling(config)
        self.attention1d_projection = Attention1dPoolingProjection(config)

    def forward(self, x, input_mask):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.attention1d_projection(x)
        return x


class FFN1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ProBLAMForSequenceClassification(ProLMModel):
    def __init__(self, config, config_path="AI4Protein/prodmm_encoder", pooling_head="mean", num_labels=1):
        super().__init__(config)
        self.num_labels = num_labels
        self.pooling_head = pooling_head
        self.prolm = ProLMModel.from_pretrained(config_path)

        if pooling_head == "mean":
            self.classifier = MeanPoolingHead(config)
        elif pooling_head == "attention":
            self.classifier = AttentionPoolingHead(config)
        elif pooling_head == "attention1d":
            self.classifier = Attention1dPoolingHead(config)
        elif pooling_head == "cls":
            self.classifier = RoFormerClassificationHead(config)
        else:
            raise NotImplementedError(f"pooling head {pooling_head} not implemented")

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize weights
        self._init_weights(self.classifier)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.prolm(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.pooling_head == "cls":
            logits = self.classifier(sequence_output)
        else:
            # mean pooling, attention pooling, attention1d pooling
            logits = self.classifier(sequence_output, attention_mask)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            print(self.config.problem_type)

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                # RuntimeError: result type Float can't be cast to the desired output type Long
                loss = loss_fct(logits, labels.float())
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == "__main__":
    # test ProBLAMForSequenceClassification
    from prodmm.encoder.configuration_prolm import ProLMConfig
    config = ProLMConfig.from_pretrained("AI4Protein/prodmm_encoder")
    model = ProBLAMForSequenceClassification(config, pooling_head="attention1d")
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    labels = torch.tensor([1, 0])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # test ProBLAMForSequenceClassification using MSE
    config.problem_type == "regression"
    labels = torch.tensor([1.0, 0.0])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
