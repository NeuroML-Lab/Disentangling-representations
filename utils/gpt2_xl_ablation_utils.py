from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers.masking_utils import *
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from torch import nn
import torch

class GPT2AblationModel(GPT2Model):
    def __init__(self, config, ablation_layers=None, sae_model=None, zero_syntax_units=False, zero_semantic_units=False):
        super().__init__(config)
        self.ablation_layers = ablation_layers
        self.sae_model = sae_model
        self.zero_syntax_units = zero_syntax_units
        self.zero_semantic_units = zero_semantic_units
        

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        self.warn_if_padding_and_no_attention_mask(input_ids, None)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        inputs_embeds = self.wte(input_ids)

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        encoder_attention_mask = None

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        kwargs.pop("position_ids", None)
        kwargs.pop("encoder_attention_mask", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("cache_position", None)
        kwargs.pop("head_mask", None)

        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states,
                None,
                cache_position,
                causal_mask,
                None,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=None,
                position_ids=position_ids,
                **kwargs,
            )[0]

            if i in self.ablation_layers:
                if self.zero_semantic_units:
                    latent_space_semantic = self.sae_model[i].linear_invertible(hidden_states)
                    latent_space_semantic[:, :, :640] = 0.0
                    hidden_states = self.sae_model[i].reconstruct(latent_space_semantic)
                if self.zero_syntax_units:
                    latent_space_syntactic = self.sae_model[i].linear_invertible(hidden_states)
                    latent_space_syntactic[:, :, 640:1280] = 0.0
                    hidden_states = self.sae_model[i].reconstruct(latent_space_syntactic)
                
                
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        past_key_values = None
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class GPT2AblationLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, ablation_layers=None, sae_model=None, zero_syntax_units=False, zero_semantic_units=False):
        super().__init__(config)
        self.transformer = GPT2AblationModel(config, ablation_layers, sae_model, zero_syntax_units, zero_semantic_units)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()