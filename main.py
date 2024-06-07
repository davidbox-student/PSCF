import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision
from torchvision.transforms import ToTensor
from torchsummary import summary
from torchviz import make_dot
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pprint import pprint
from torch.utils.data import DataLoader


class PetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=4, **kwargs
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        # for image segmentation dice loss could be the best first choice
        self.predicted_masks = []
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h = image.shape[2]
        w = image.shape[3]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        assert mask.ndim == 4 and mask.shape[1] == 4

        # Check that mask values in between 0 and 1
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        self.store_predicted_masks(pred_mask)
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multilabel", num_classes=4)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def store_predicted_masks(self, pred_masks):
        self.predicted_masks.extend(pred_masks)

    def get_predicted_masks(self):
        return torch.stack(self.predicted_masks).long()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def train_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


class MyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, decision=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_filenames = os.listdir(img_dir)
        self.mask_filenames = os.listdir(mask_dir)
        self.decision = decision

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        # Split mask into individual channels
        if self.decision:
            width, height = img.size
            # Set the fragment size
            fragment_size = 320
            # Calculate the maximum starting position for the fragment
            max_start_x = width - fragment_size
            max_start_y = height - fragment_size
            # Generate random starting position for the fragment
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(0, max_start_y)
            # Crop the image to get the fragment
            img = img.crop((start_x, start_y, start_x + fragment_size, start_y + fragment_size))
            mask = mask.crop((start_x, start_y, start_x + fragment_size, start_y + fragment_size))

        mask = np.array(mask)
        mask = mask / 255.0
        mask = np.array(Image.fromarray(mask.astype('uint8'), 'RGB'))

        mask_tensor = torch.zeros((img.size[1], img.size[0], 4))
        mask_tensor[:, :, 0] = torch.as_tensor(mask[:, :, 0])
        mask_tensor[:, :, 1] = torch.as_tensor(mask[:, :, 1])
        mask_tensor[:, :, 2] = torch.as_tensor(mask[:, :, 2])
        mask_tensor[:, :, 3] = torch.as_tensor(mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])
        mask_tensor = torch.tensor(np.array(mask_tensor)).permute(2, 0, 1)

        # Convert image to tensor
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        # Apply transformation
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor


def ShowResults(model=None, load_model=None, decision=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_model:
        if decision == 'Augmentation':
            main_model = torch.load('./best_model_Augmentation.pth')
        else:
            main_model = torch.load('./best_model.pth')
    else:
        main_model = model

    model.to(device)
    masks_folder = "DataBaseMasks"
    masks_filenames = os.listdir(masks_folder)
    # Define a transform to convert image files to tensors
    transform = ToTensor()
    masks = []
    for image_path in masks_filenames:
        mask = Image.open(masks_folder + '/' + image_path).convert('RGB')
        mask = np.array(mask)
        mask = mask / 255.0
        mask = np.array(Image.fromarray(mask.astype('uint8'), 'RGB'))
        if decision == 'Augmentation':
            mask_tensor = torch.zeros((320, 320, 4))
        else:
            mask_tensor = torch.zeros((480, 480, 4))
        mask_tensor[:, :, 0] = torch.as_tensor(mask[:, :, 0])
        mask_tensor[:, :, 1] = torch.as_tensor(mask[:, :, 1])
        mask_tensor[:, :, 2] = torch.as_tensor(mask[:, :, 2])
        mask_tensor[:, :, 3] = torch.as_tensor(mask[:, :, 0] & mask[:, :, 1] & mask[:, :, 2])
        mask_tensor = mask_tensor.permute(2, 0, 1)
        masks.append(mask_tensor)

    masks_batch = torch.stack(masks).long()

    with torch.no_grad():
        main_model.eval()
        if load_model:
            output = main_model(masks_batch)
        else:
            output = main_model.get_predicted_masks()[:210]

    # Compute evaluation metrics
    tp, fp, fn, tn = smp.metrics.get_stats(output, masks_batch, mode='multilabel', threshold=0.5)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    # Print the evaluation metrics
    print("Evaluation Metrics:")
    print("IoU Score:", iou_score)
    print("F1 Score:", f1_score)
    print("F2 Score:", f2_score)
    print("Accuracy:", accuracy)
    print("Recall:", recall)


def main():
    # download data
    root = "."
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    teach_model = input('Teach model? (Y/N):')
    if teach_model == 'n':
        decision2 = input('Normal or Augmentation : ')
    else:
        decision = input('Normal or Augmentation : ')
        echos = input('Number of Epochs : ')
        showdata = input('Show results after? (Y): ')
        if showdata == 'y':
            decision2 = input('Normal or Augmentation : ')
        if decision == 'Normal':
            img_dataset = MyDataset("DataBaseImages", "DataBaseMasks", transform=None)
            mask_dataset = MyDataset("DataBaseImages", "DataBaseMasks", transform=None)
        if decision == 'Augmentation':
            img_dataset = MyDataset("Images", "Masks", transform=None, decision=True)
            mask_dataset = MyDataset("Images", "Masks", transform=None, decision=True)
        if decision != 'Normal' and decision != 'Augmentation':
            exit(2)
        n_cpu = 1
        train_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=True, num_workers=n_cpu)
        valid_dataloader = DataLoader(mask_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)
        # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    model = PetModel("FPN", "resnet34", in_channels=3)
    if teach_model == 'n':
        ShowResults(model, True, decision2)
    else:
        trainer = pl.Trainer(
            max_epochs=int(echos)
        )
        trainer.fit(
            model,
            train_dataloader,
            valid_dataloader,
        )

        # # run validation dataset
        valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
        pprint(valid_metrics)
        if decision == 'Augmentation':
            torch.save(model, './best_model_Augmentation.pth')
        if decision == 'Normal':
            torch.save(model, './best_model_Normal.pth')
        if showdata == 'y':
            ShowResults(model, False, decision2)


if __name__ == "__main__":
    main()
