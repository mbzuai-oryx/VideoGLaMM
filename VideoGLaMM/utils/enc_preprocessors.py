import torch
import numpy as np
from transformers import CLIPImageProcessor
from model.videogpt_plus.model.internvideo.utils import VideoTrainProcessor

class CLIPPreprocessorL14_224:
    def __init__(self, clip_model):
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def preprocess(self, pil_images):
        '''
        Args:
            pil_images: 
                list of PIL images or numpy arrays (if video)
                a single PIL image or numpy array (if image)
            
        Returns:
            image_clip: Tx(3, 224, 224) if video, (3, 224, 224) if image
        '''
        
        if isinstance(pil_images, list): # If video
            image_clip = [self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in pil_images]        
        else: # If image
            image_clip = self.image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"][0]
        
        return image_clip
    
class CLIPPreprocessorL14_336:
    def __init__(self, clip_model):
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    def preprocess(self, pil_images):
        '''
        Args:
            pil_images: 
                list of PIL images or numpy arrays (if video)
                a single PIL image or numpy array (if image)
            
        Returns:
            image_clip: Tx(3, 336, 336) if video, (3, 336, 336) if image
        '''
        
        if isinstance(pil_images, list): # If video
            image_clip = [self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in pil_images]        
        else: # If image
            image_clip = self.image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"][0]
        
        return image_clip
    
class InternVideo2Preprocessor:
    def __init__(self, clip_model):
        self.video_processor = VideoTrainProcessor()

    def preprocess(self, pil_images):
        '''
        Args:
            pil_images: 
                list of PIL images or numpy arrays (if video)
                a single PIL image or numpy array (if image)
            
        Returns:
            ???
        '''
        
        patch_images = self.video_processor.preprocess(patch_images)['pixel_values']
        
        image = self.video_processor.preprocess([np.array(image)], use_image=True)['pixel_values'][0]
        
        return image_clip
    


from model.chatunivi import constants as constants_chatunivi
class EncPreprocessor_ChatUniVi:
    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.num_frames = constants_chatunivi.MAX_IMAGE_LENGTH
        
        self.frame_resolution_clip = 224
        
    def preprocess(self, pil_images):
        '''
        Args:
            pil_images: 
                list of PIL images or numpy arrays (if video)
                a single PIL image or numpy array (if image)
            
        Returns:
            image_clip: Tx(3, 224, 224) if video, (3, 224, 224) if image
        '''
        
        if isinstance(pil_images, list): # If video
            image_clip = [self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in pil_images]        
        else: # If image
            image_clip = self.image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"][0]
        
        # return image_clip
        data_dict = {}
        data_dict['images'] = image_clip
        data_dict['context_images'] = None
        return data_dict
    

from model.videogpt_plus import constants as constants_videogpt_plus
class EncPreprocessor_VideoGPTPlus:
    def __init__(self):
        # self.num_frames_iv = constants_videogpt_plus.NUM_FRAMES
        # self.num_frames_clip = constants_videogpt_plus.NUM_CONTEXT_IMAGES
        self.num_frames = constants_videogpt_plus.NUM_FRAMES
        
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.video_processor = VideoTrainProcessor(num_frames = self.num_frames)
        
        
        
        self.frame_resolution_iv = 224
        self.frame_resolution_clip = 336 # self.image_processor.crop_size['height']
        
    def preprocess(self, pil_images):
        '''
        Args:
            pil_images: 
                list of PIL images or numpy arrays (if video)
                a single PIL image or numpy array (if image)
            
        Returns:
            ???
        '''
        
        data_dict = {}
        # if 'image'
        if not isinstance(pil_images, list):

            image = pil_images
            preprocessed_image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            data_dict['images'] = preprocessed_image
            data_dict['context_images'] = None

        # elif "video"
        elif isinstance(pil_images, list):
            
            # make sure the video length is 'num_frames'
            if len(pil_images) > self.num_frames:
                pil_images = pil_images[:self.num_frames]
            elif len(pil_images) < self.num_frames:
                while len(pil_images) < self.num_frames:
                    pil_images.append(pil_images[-1])
                        
            video_frames = self.video_processor.preprocess(pil_images)['pixel_values']
            context_images = [self.image_processor.preprocess(im, return_tensors='pt')['pixel_values'][0] for im in pil_images]
            
            # padding
            if len(context_images) < self.num_frames:
                while len(context_images) < self.num_frames:
                    context_images.append(torch.zeros((3, self.image_processor.crop_size['height'], self.image_processor.crop_size['width'])))

            if len(video_frames) < self.num_frames:
                while len(video_frames) < self.num_frames:
                    video_frames.append(torch.zeros((3, self.frame_resolution_iv, self.frame_resolution_iv)))

            data_dict['images'] = video_frames
            data_dict['context_images'] = context_images
        
        return data_dict
        

            