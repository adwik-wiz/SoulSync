import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
from PIL import Image

class EmoSet(Dataset):
    ATTRIBUTES_MULTI_CLASS = [
        'scene', 'facial_expression', 'human_action', 'brightness', 'colorfulness',
    ]
    ATTRIBUTES_MULTI_LABEL = [
        'object'
    ]
    NUM_CLASSES = {
        'brightness': 11,
        'colorfulness': 11,
        'scene': 254,
        'object': 409,
        'facial_expression': 6,
        'human_action': 264,
    }

    def __init__(self, data_root, num_emotion_classes, phase):
        assert num_emotion_classes in (6, 2)
        assert phase in ('train', 'val', 'test')
        self.transforms_dict = self.get_data_transforms()

        self.info = self.get_info(data_root, num_emotion_classes)

        if phase == 'train':
            self.transform = self.transforms_dict['train']
        elif phase == 'val':
            self.transform = self.transforms_dict['val']
        elif phase == 'test':
            self.transform = self.transforms_dict['test']
        else:
            raise NotImplementedError

        try:
            data_store = json.load(open(os.path.join(data_root, f'{phase}.json')))
            self.data_store = [
                [
                    self.info['emotion']['label2idx'][item[0]],
                    json.load(open(os.path.join(data_root, item[2])))["image_id"],
                    os.path.join(data_root, item[1]),
                    os.path.join(data_root, item[2])
                ]
                for item in data_store
                if item[0] in ["amusement", "awe", "contentment", "excitement", "anger", "sadness"]
            ]
            print(f"Data store loaded successfully. Total samples: {len(self.data_store)}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            self.data_store = []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            self.data_store = []

    @classmethod
    def get_data_transforms(cls):
        transforms_dict = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return transforms_dict

    def get_info(self, data_root, num_emotion_classes):
        assert num_emotion_classes in (6, 2)
        try:
            info = json.load(open(os.path.join(data_root, 'info.json')))
        except FileNotFoundError as e:
            print(f"Info file not found: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding info.json: {e}")
            return {}

        if num_emotion_classes == 6:
            pass
        elif num_emotion_classes == 2:
            emotion_info = {
                'label2idx': {
                    'amusement': 0,
                    'awe': 0,
                    'contentment': 0,
                    'excitement': 0,
                    'anger': 1,
                    'sadness': 1,
                },
                'idx2label': {
                    '0': 'positive',
                    '1': 'negative',
                }
            }
            info['emotion'] = emotion_info
        else:
            raise NotImplementedError

        return info

    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image

    def load_annotation_by_path(self, path):
        json_data = json.load(open(path))
        return json_data

    def __getitem__(self, item):
        emotion_label_idx, image_id, image_path, annotation_path = self.data_store[item]
        image = self.load_image_by_path(image_path)
        annotation_data = self.load_annotation_by_path(annotation_path)
        data = {'image_id': image_id, 'image': image, 'emotion_label_idx': emotion_label_idx}

        for attribute in self.ATTRIBUTES_MULTI_CLASS:
            attribute_label_idx = -1
            if attribute in annotation_data:
                attribute_label_idx = self.info[attribute]['label2idx'][str(annotation_data[attribute])]
            data.update({f'{attribute}_label_idx': attribute_label_idx})

        for attribute in self.ATTRIBUTES_MULTI_LABEL:
            assert attribute == 'object'
            num_classes = self.NUM_CLASSES[attribute]
            attribute_label_idx = torch.zeros(num_classes)
            if attribute in annotation_data:
                for label in annotation_data[attribute]:
                    attribute_label_idx[self.info[attribute]['label2idx'][label]] = 1
            data.update({f'{attribute}_label_idx': attribute_label_idx})

        return data

    def __len__(self):
        return len(self.data_store)


if __name__ == '__main__':
    print("Starting EmoSet script...")

    data_root = r'D:\EmoSet-118K'
    num_emotion_classes = 6
    phase = 'test'

    try:
        print(f"Initializing dataset with data root: {data_root}, phase: {phase}")
        dataset = EmoSet(
            data_root=data_root,
            num_emotion_classes=num_emotion_classes,
            phase=phase,
        )
        print(f"Dataset initialized successfully with {len(dataset)} samples.")

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        print("DataLoader initialized successfully.")

        for i, data in enumerate(dataloader):
            print(f"{i}    {data['emotion_label_idx']}")

        print("Script completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
