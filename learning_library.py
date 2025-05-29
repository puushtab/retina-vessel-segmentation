import os
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

# --- U-Net Components ---
class DoubleConv(nn.Module):
    """Applies two consecutive convolutional layers with ReLU and BatchNorm."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """Defines the U-Net architecture."""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        print("[INFO] Initializing U-Net architecture...")

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder with skip connections
        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return torch.sigmoid(self.final(dec1))

# --- Dataset Loader with Augmentation ---
class RetinaSegDataset(Dataset):
    def __init__(self, image_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.transform = transform
        self.augment = augment

        # Find all star*.jpg images
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.startswith('star') and f.endswith('.jpg')])

        # Create corresponding GT filenames
        self.gt_files = []
        for f in self.image_files:
            # Extract number from filename (e.g., star01_OSC.jpg -> 01)
            num = ''.join(filter(str.isdigit, f.split('_')[0]))  # Get digits from first part
            if num:
                gt_name = f'GT_{num}.png'  # Use extracted number as-is
                gt_path = os.path.join(image_dir, gt_name)
                if os.path.exists(gt_path):  # Only add if GT file exists
                    self.gt_files.append(gt_name)
                else:
                    print(f"[WARNING] GT file not found: {gt_name}")
                    self.image_files.remove(f)  # Remove image if no corresponding GT

        print(f"[INFO] Found {len(self.image_files)} image-mask pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.image_dir, self.gt_files[idx])

        # Load and convert images
        image = Image.open(img_path).convert('L')
        mask = Image.open(gt_path).convert('L')

        # Resize to model input size
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        mask = mask.resize((256, 256), Image.Resampling.NEAREST)  # Use nearest for binary masks

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # Add rotation augmentation
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Convert to tensors and normalize
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image to [0, 1] and ensure mask is binary
        mask = (mask > 0.5).float()  # Convert to binary mask

        return image, mask


def infer_segmentation(img, img_mask, model_path, seuil=0.5, device='cpu'):
    """
    Load trained U-Net model and run inference on input image.
    """
    print(f"[INFO] Loading model from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("[INFO] Please train the model first by calling train_model()")
        return np.zeros(img.shape, dtype=np.uint8)
    
    # Load model
    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return np.zeros(img.shape, dtype=np.uint8)

    orig_shape = img.shape
    print(f"[INFO] Original image shape: {orig_shape}")

    # Preprocess input image
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)  # Convert to grayscale if RGB
    
    img_pil = Image.fromarray(img.astype(np.uint8)).convert('L').resize((256, 256))
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    print("[INFO] Running inference...")
    with torch.no_grad():
        output = model(img_tensor)
        prob = output.squeeze().cpu().numpy()

    print(f"[DEBUG] Output probability range: [{prob.min():.3f}, {prob.max():.3f}]")

    # Resize img_mask to match probability map
    if img_mask.shape != (256, 256):
        img_mask_pil = Image.fromarray(img_mask.astype(np.uint8) * 255).resize((256, 256))
        img_mask_resized = np.array(img_mask_pil) > 127
    else:
        img_mask_resized = img_mask.astype(bool)

    # Apply threshold and mask
    pred_mask = (prob > seuil) & img_mask_resized

    # Convert to uint8 0/255
    pred_mask = (pred_mask.astype(np.uint8)) * 255

    # Resize back to original shape
    if pred_mask.shape != orig_shape:
        pred_mask_pil = Image.fromarray(pred_mask).resize((orig_shape[1], orig_shape[0]), 
                                                         Image.Resampling.NEAREST)
        pred_mask = np.array(pred_mask_pil)

    print(f"[INFO] Inference complete. Threshold: {seuil}")
    print(f"[INFO] Pixels above threshold: {np.sum(pred_mask > 0)}")
    print(f"[INFO] Output mask shape: {pred_mask.shape}")
    
    return pred_mask


# --- Improved Training Loop ---
def train_model(epochs=50, batch_size=4, learning_rate=1e-4, save_path='unet_retina.pth'):
    """
    Trains the U-Net on retinal images with improved training strategy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Initialize model
    model = UNet().to(device)
    
    # Prepare dataset
    print("[INFO] Preparing dataset...")
    dataset = RetinaSegDataset('images_IOSTAR', augment=True)
    
    if len(dataset) == 0:
        print("[ERROR] No valid image-mask pairs found!")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Use Dice loss + BCE for better vessel segmentation
    def dice_loss(pred, target, smooth=1):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def combined_loss(pred, target):
        bce = F.binary_cross_entropy(pred, target)
        dice = dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice

    print(f"[INFO] Starting training for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for i, (img, mask) in enumerate(dataloader):
            img, mask = img.to(device), mask.to(device)
            
            # Forward pass
            output = model(img)
            loss = combined_loss(output, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            if (i + 1) % max(1, len(dataloader) // 4) == 0:
                avg_loss = running_loss / num_batches
                print(f"[DEBUG] Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)
        
        print(f"[INFO] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] New best model saved with loss: {best_loss:.4f}")

    print(f"[INFO] Training complete! Best model saved to '{save_path}'")
    return model


# --- Example Usage ---
if __name__ == '__main__':
    # Train the model
    print("=" * 50)
    print("TRAINING PHASE")
    print("=" * 50)
    model = train_model(epochs=30, batch_size=2)
    
    # Test inference
    print("\n" + "=" * 50)
    print("INFERENCE PHASE")
    print("=" * 50)
    
    # Load test image
    test_img_path = './images_IOSTAR/star01_OSC.jpg'
    if os.path.exists(test_img_path):
        test_img = np.array(Image.open(test_img_path).convert('L'))
        
        # Create circular mask
        nrows, ncols = test_img.shape
        row, col = np.ogrid[:nrows, :ncols]
        test_mask = np.ones(test_img.shape, dtype=bool)
        invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows/2)**2)
        test_mask[invalid_pixels] = False
        
        # Run inference
        pred = infer_segmentation(test_img, test_mask, 'unet_retina.pth', seuil=0.3)
        
        # Save result
        result_img = Image.fromarray(pred)
        result_img.save('test_segmentation_result.png')
        print("[INFO] Test segmentation saved as 'test_segmentation_result.png'")
    else:
        print(f"[ERROR] Test image not found: {test_img_path}")