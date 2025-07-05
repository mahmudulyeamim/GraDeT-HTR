import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

from config import DTrOCRConfig
from processor import DTrOCRProcessor

class HandwrittenDataset(Dataset):
    def __init__(self, images_dir, data_frame, config: DTrOCRConfig):
        super(HandwrittenDataset, self).__init__()
        
        self.images_dir = images_dir
        self.df = data_frame
        
        self.image_ids = self.df["image_id"].values
        self.texts = self.df["text"].values
        
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.images_dir, str(self.image_ids[index]))).convert('RGB')
        text = str(self.texts[index])
        
        inputs = self.processor(
            images=image,
            texts=text,
            padding=True,
            return_tensors="pt",
            return_labels=True,
        )
        
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels': inputs.labels[0]
        }

def split_data(images_dir, labels_file, config, test_size=0.05, random_seed=42):
    df = pd.read_csv(labels_file)
    # Split into train + validation/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    train_dataset = HandwrittenDataset(images_dir, train_df, config)
    test_dataset = HandwrittenDataset(images_dir, test_df, config)
    
    return train_dataset, test_dataset
