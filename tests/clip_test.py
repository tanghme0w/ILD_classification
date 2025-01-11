from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
from PIL import Image
import torch.nn.functional as F


MODEL_PATH = '/tanghaomiao/medai/clip-vit-large-patch14'

model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to('cuda')
processor = CLIPImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

image = Image.open('test.png')
image = processor(image, return_tensors='pt').to('cuda')
image_features = model.get_image_features(**image)
image_features = F.normalize(image_features, dim=1)

text = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird', 'a photo of a CT']
text_ids = tokenizer(text, return_tensors='pt', padding=True).to('cuda')
text_features = model.get_text_features(**text_ids)
text_features = F.normalize(text_features, dim=1)

# compute the similarity between image_features and text_features
similarity = image_features @ text_features.T

for i in range(len(text)):
    print(f'{text[i]}: {similarity[0][i]}')
