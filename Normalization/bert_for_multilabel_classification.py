import scipy
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification
import torch


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for multilabel classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
       The primary change here is the usage of Binary cross-entropy with logits (BCEWithLogitsLoss) loss function
    instead of vanilla cross-entropy loss (CrossEntropyLoss) that is used for multiclass classification.
    Binary cross-entropy loss allows our model to assign independent probabilities to the labels.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, config.num_labels),  torch.nn.Sigmoid())

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #sigmoid_logits = [[scipy.special.expit(nb) for nb in out] for out in logits]


        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:

                labels = labels.type_as(logits)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)