import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig

class UnifiedModel(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, num_labels=2, contrastive_embedding_size=128):
        super(UnifiedLlamaModel, self).__init__(config)
        # Define layers for ranking and classification tasks
        self.fc_rank = nn.Linear(config.hidden_size, contrastive_embedding_size)
        self.fc_classifier = nn.Linear(contrastive_embedding_size, num_labels)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(config.hidden_size)  # Added LayerNorm to normalize hidden states
        self.bias_removal = None  # Initialize bias removal as None, can be set later with set_bias_removal

    def set_bias_removal(self, bias_direction):
        """ Set the bias direction for debiasing embeddings """
        self.bias_removal = BiasRemoval(bias_direction)
    
    def forward(self, input_ids, attention_mask, bias_input_ids=None, bias_attention_mask=None, task="qa", **kwargs):
        # Forward pass for QA task, inheriting from LlamaForCausalLM
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=True, **kwargs)
        
        if task == "qa":
            # Process outputs for question-answering task
            logits = outputs.logits
            if self.bias_removal is not None:
                # Debias hidden states if bias_removal is set
                hidden_states = outputs.hidden_states[-1]  # Take the last hidden states
                hidden_states = self.bias_removal(hidden_states)
                # Recompute logits with debiased hidden states
                logits = super().forward(hidden_states=hidden_states).logits
            return logits

        elif task == "contrastive":
            # For contrastive learning: comparing biased and unbiased inputs
            bias_outputs = super().forward(input_ids=bias_input_ids, attention_mask=bias_attention_mask, output_hidden_states=True, **kwargs)
            bias_hidden_states = bias_outputs.hidden_states[-1]
            main_hidden_states = outputs.hidden_states[-1]

            # Apply LayerNorm to stabilize hidden states
            main_hidden_states = self.layer_norm(main_hidden_states)
            bias_hidden_states = self.layer_norm(bias_hidden_states)

            # Clamp hidden states to prevent extreme values
            main_hidden_states = torch.clamp(main_hidden_states, min=-1e2, max=1e2)
            bias_hidden_states = torch.clamp(bias_hidden_states, min=-1e2, max=1e2)

            # Compute embeddings for contrastive task
            main_projection = self.fc_rank(main_hidden_states)
            bias_projection = self.fc_rank(bias_hidden_states)

            return main_projection, bias_projection

class BiasRemoval(nn.Module):
    def __init__(self, bias_direction):
        super(BiasRemoval, self).__init__()
        # Normalize bias direction vector
        self.bias_direction = bias_direction / torch.norm(bias_direction, p=2)

    def forward(self, embeddings):
        # Remove bias by projecting embeddings away from the bias direction
        bias_projection = torch.matmul(embeddings, self.bias_direction.unsqueeze(-1)) * self.bias_direction.unsqueeze(0)
        adjusted_embeddings = embeddings - bias_projection
        return adjusted_embeddings

# Custom contrastive loss function for contrastive learning tasks
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate pairwise distances between anchor-positive and anchor-negative pairs
        pos_dist = F.pairwise_distance(anchor, positive) + 1e-8
        neg_dist = F.pairwise_distance(anchor, negative) + 1e-8
        # Loss is computed as margin-based hinge loss
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss
