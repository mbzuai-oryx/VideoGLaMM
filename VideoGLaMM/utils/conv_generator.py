import re

class ConvGenerator_Base:
    DEFAULT_VIDEO_TOKEN = None
    DEFAULT_IMAGE_TOKEN = None
    DEFAULT_IM_START_TOKEN = None
    DEFAULT_IM_END_TOKEN = None
    DEFAULT_VID_START_TOKEN = None
    DEFAULT_VID_END_TOKEN = None
    NUM_FRAMES = None
    IGNORE_INDEX = None
    
    def __init__(self, use_mm_start_end):
        self.conversation_lib = None
        self.tokenizer_image_token = None
        
        self.use_mm_start_end = use_mm_start_end
    
    
    def __preprocess_multimodal(self, source):

        for sentence in source:
        
            # If the <video> or <image> token is at the end of the sentence, this function will move it to the beginning
            if self.DEFAULT_VIDEO_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(self.DEFAULT_VIDEO_TOKEN, "").strip()
                )
                sentence["value"] = self.DEFAULT_VIDEO_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            if self.DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(self.DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = self.DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            
            # If the <video> or <image> token is in the middle of the sentence, this function will replace it with <vid_start><image>...<vid_end> or <im_start><image><im_end>
            im_replace_token, vid_replace_token = self.DEFAULT_IMAGE_TOKEN, self.DEFAULT_IMAGE_TOKEN * self.NUM_FRAMES 
            if self.use_mm_start_end: 
                im_replace_token = (self.DEFAULT_IM_START_TOKEN + im_replace_token + self.DEFAULT_IM_END_TOKEN) # replace <image> token with <im_start><image><im_end>
                vid_replace_token = self.DEFAULT_VID_START_TOKEN + vid_replace_token + self.DEFAULT_VID_END_TOKEN # replace <video> token with <vid_start><image>...<vid_end>
            sentence["value"] = sentence["value"].replace(self.DEFAULT_IMAGE_TOKEN, im_replace_token)
            sentence["value"] = sentence["value"].replace(self.DEFAULT_VIDEO_TOKEN, vid_replace_token)
            
        return source
    
    def apply(self, source):
        '''
        Args:
            source: list of dict
            e.g.
                [{'from': 'human', 'value': '<video> \n What is the name of the cat?'},
                {'from': 'gpt', 'value': 'The name of the cat is Fluffy.'}]
        '''
        conv = self.default_conversation.copy()

        source = self.__preprocess_multimodal(source)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"role:{role} is not the same as {conv.roles[j % 2]}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        
        return conversations
        
    def apply_on_semseg_dataset(self, source):
        
        conversations = []
        conv = self.default_conversation.copy()
        
        for i in range(len(source)):
            conv.messages = []
            conv.append_message(conv.roles[0], source[i]['from'])
            conv.append_message(conv.roles[1], source[i]['value'])
            conversations.append(conv.get_prompt())
            
        return conversations
    
    def apply_for_chat(self, prompt_text, type='video', tokenizer=None):
        ''' 
        Returns: 
        - input_ids: tensor of text prompt tokenized : (B x L)
        '''
        # if Video
        if type == 'video':
            
            # Add video token to text prompt
            prompt = self.DEFAULT_VIDEO_TOKEN + "\n" + prompt_text
            replace_token, vid_replace_token = self.DEFAULT_IMAGE_TOKEN, self.DEFAULT_IMAGE_TOKEN * self.NUM_FRAMES  #NOTE: The value given here for MAX_IMAGE_LENGTH does not matter due to the way how it is handled in In arch.py. It only needs to be a value greater than 2
            if self.use_mm_start_end: 
                replace_token = (self.DEFAULT_IM_START_TOKEN + replace_token + self.DEFAULT_IM_END_TOKEN) # replace <image> token with <im_start><image><im_end>
                vid_replace_token = self.DEFAULT_VID_START_TOKEN + vid_replace_token + self.DEFAULT_VID_END_TOKEN # replace <video> token with <vid_start><image>...<vid_end>
            prompt = prompt.replace(self.DEFAULT_IMAGE_TOKEN, replace_token)
            prompt = prompt.replace(self.DEFAULT_VIDEO_TOKEN, vid_replace_token)
            # Create conversation object
            conv = self.default_conversation.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            
            # Tokenize conversation prompt
            input_ids = self.tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()
            
        #  if Image
        elif type == 'image':
            # Add image token to text prompt
            prompt = self.DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
            if self.use_mm_start_end:
                replace_token = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(self.DEFAULT_IMAGE_TOKEN, replace_token)

            # Create conversation object
            conv = self.default_conversation.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            # Tokenize conversation prompt
            input_ids = self.tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()
        
        return input_ids
        
        

from model.chatunivi import conversation as conversation_lib_chatunivi
from model.chatunivi import constants as constants_chatunivi
from model.chatunivi.mm_utils import tokenizer_image_token as tokenizer_image_token_chatunivi

class ConvGenerator_ChatUniVi(ConvGenerator_Base):
    DEFAULT_VIDEO_TOKEN = constants_chatunivi.DEFAULT_VIDEO_TOKEN
    DEFAULT_IMAGE_TOKEN = constants_chatunivi.DEFAULT_IMAGE_TOKEN
    DEFAULT_IM_START_TOKEN = constants_chatunivi.DEFAULT_IM_START_TOKEN
    DEFAULT_IM_END_TOKEN = constants_chatunivi.DEFAULT_IM_END_TOKEN
    DEFAULT_VID_START_TOKEN = constants_chatunivi.DEFAULT_VID_START_TOKEN
    DEFAULT_VID_END_TOKEN = constants_chatunivi.DEFAULT_VID_END_TOKEN
    IGNORE_INDEX = constants_chatunivi.IGNORE_INDEX
    NUM_FRAMES = constants_chatunivi.MAX_IMAGE_LENGTH
    
    def __init__(self, use_mm_start_end):
        super().__init__(use_mm_start_end=use_mm_start_end)
        
        # self.conversation_lib = conversation_lib_chatunivi
        self.default_conversation = conversation_lib_chatunivi.conv_templates["llava_v1"]
        self.tokenizer_image_token = tokenizer_image_token_chatunivi
                
    def preprocess_fn(self, conversation_list, targets, tokenizer=None):
        ''' preprocess_v1 '''
                
        conv = self.default_conversation.copy()
        sep = conv.sep + conv.roles[1] + ": "
        # Mask targets
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            
            rounds = conversation.split(conv.sep2)
            cur_len = 1
            
            target[:cur_len] = self.IGNORE_INDEX # ignore 0'th position of the target in loss calculation (i.e. [BOS] token)
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep) # split by ' ASSISTANT: '
                
                ## Assert only one question-answer pair is in the selected sample
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep # add ' ASSISTANT: ' back to the question

                if self.DEFAULT_IMAGE_TOKEN in conversation: 
                    round_len = len(self.tokenizer_image_token(rou, tokenizer))                  # length of the conversation round (question and answer)
                    instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 2   # length of the question only # Why subtract 2 ???
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = self.IGNORE_INDEX # ignore the part of the target corresponding to the instruction(question) in the loss calculation

                cur_len += round_len
            target[cur_len:] = self.IGNORE_INDEX # ignore the remaining parts of the target in the loss calculation

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = self.IGNORE_INDEX
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." , f" (ignored)")
    

from model.videogpt_plus import conversation as conversation_lib_videogpt_plus
from model.videogpt_plus import constants as constants_videogpt_plus
from model.videogpt_plus.mm_utils import tokenizer_image_token as tokenizer_image_token_videogpt_plus

class ConvGenerator_VideoGPTPlus(ConvGenerator_Base):
    DEFAULT_VIDEO_TOKEN = constants_videogpt_plus.DEFAULT_VIDEO_TOKEN
    DEFAULT_IMAGE_TOKEN = constants_videogpt_plus.DEFAULT_IMAGE_TOKEN
    DEFAULT_IM_START_TOKEN = constants_videogpt_plus.DEFAULT_IM_START_TOKEN
    DEFAULT_IM_END_TOKEN = constants_videogpt_plus.DEFAULT_IM_END_TOKEN
    DEFAULT_VID_START_TOKEN = constants_videogpt_plus.DEFAULT_VID_START_TOKEN
    DEFAULT_VID_END_TOKEN = constants_videogpt_plus.DEFAULT_VID_END_TOKEN
    IGNORE_INDEX = constants_videogpt_plus.IGNORE_INDEX
    NUM_FRAMES = constants_videogpt_plus.NUM_FRAMES
    
    def __init__(self, use_mm_start_end, base_type='phi3'):
        super().__init__(use_mm_start_end=use_mm_start_end)
        
        self.base_type = base_type
        
        if base_type == 'phi3':
            self.default_conversation = conversation_lib_videogpt_plus.conv_templates["phi3_instruct"]
        elif base_type == 'llama3_1':
            self.default_conversation = conversation_lib_videogpt_plus.conv_templates["llama3_1"]
        
        self.tokenizer_image_token = tokenizer_image_token_videogpt_plus
        
        
    def preprocess_fn(self, conversation_list, targets, tokenizer=None):
        if self.base_type == 'phi3':
            self.preprocess_fn_phi3(conversation_list, targets, tokenizer)
        elif self.base_type == 'llama3_1':
            self.preprocess_fn_llama3_1(conversation_list, targets, tokenizer)
            
    def preprocess_fn_phi3(self, conversation_list, targets, tokenizer=None):
        ''' preprocess_phi3 '''
                            
        conv = self.default_conversation.copy()
        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
            cur_len = 0
            target[:cur_len] = self.IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if self.DEFAULT_IMAGE_TOKEN in conversation: 
                    round_len = len(self.tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                if i == 0:
                    round_len += 1
                    instruction_len += 1
                else:
                    round_len -= 2
                    instruction_len -= 2

                target[cur_len: cur_len + instruction_len] = self.IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = self.IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = self.IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
    
    def preprocess_fn_llama3_1(self, conversation_list, targets, tokenizer=None):
        ''' preprocess_llama3_1 '''
        
        offset = 0 if targets[0][0] != tokenizer.bos_token_id else 1
        
        conv = self.default_conversation.copy()
        # assert conv.sep_style == conversation_lib_videogpt_plus.SeparatorStyle.LLAMA_3_1
        assert conv.sep_style == conversation_lib_videogpt_plus.SeparatorStyle.TWO
        # Mask targets
        # sep = conv.sep + conv.roles[1]
        # sep= '<|start_header_id|>' + conv.roles[1] + '<|end_header_id|>' + '\n\n'
        sep = conv.sep + conv.roles[1] + ":"
        
        
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = self.IGNORE_INDEX
            
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if self.DEFAULT_IMAGE_TOKEN in conversation: 
                    round_len = len(self.tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - offset
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - offset

                target[cur_len : cur_len + instruction_len] = self.IGNORE_INDEX

                cur_len += round_len + (1 - offset) #starting from index 0, then cur_len will not cover eos token
        
            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = self.IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                # else:
                #     print(f"SUCCESS: {cur_len} and {total_len}.")
