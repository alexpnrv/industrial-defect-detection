class DefectDataset(Dataset):
    def __init__(self, images_dir, annotations, transform=None):
        self.images_dir = images_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, str(self.annotations[idx]['filename']))
        image = Image.open(img_path).convert('RGB')
        label = self.annotations[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label