from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, Phi3Model, Phi3Config, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

@dataclass
class CausalLMOutputWithPastAndSegTokenMask(CausalLMOutputWithPast):
    seg_token_mask: Optional[torch.FloatTensor] = None
    

from ..arch import MetaModel, VideoGPTPlusMetaForCausalLM


class VideoGPTPlusPhi3Config(Phi3Config):
    model_type = "VideoGPT+"


class VideoGPTPlusPhi3Model(MetaModel, Phi3Model):
    config_class = VideoGPTPlusPhi3Config

    def __init__(self, config: Phi3Config):
        super(VideoGPTPlusPhi3Model, self).__init__(config)


class VideoGPTPlusPhi3ForCausalLM(Phi3ForCausalLM, VideoGPTPlusMetaForCausalLM):
    config_class = VideoGPTPlusPhi3Config

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = VideoGPTPlusPhi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            context_images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        (
            input_ids, 
            attention_mask, 
            past_key_values, 
            inputs_embeds, 
            labels,
            new_seg_token_masks
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, context_images)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable videogpt_plus/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPastAndSegTokenMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
            seg_token_mask=new_seg_token_masks
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            images=None, context_images=None, **kwargs
    ):
        '''This function will be called from the ChatUniViLlamaForCausalLM.generate() function just before __call__()/forward() function '''
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # "images": kwargs.get("images", None),
                # "context_images": kwargs.get("context_images", None),
                "images": images,
                "context_images": context_images ,
            }
        )
        return model_inputs


AutoConfig.register("VideoGPT+", VideoGPTPlusPhi3Config)
AutoModelForCausalLM.register(VideoGPTPlusPhi3Config, VideoGPTPlusPhi3ForCausalLM)
