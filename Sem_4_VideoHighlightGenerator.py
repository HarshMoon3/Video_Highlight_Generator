import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import librosa
import whisper
import warnings
import yt_dlp
from torch.utils.data import Dataset, DataLoader
import json
import glob
# UPDATED IMPORTS: Direct imports to fix "Unresolved reference" errors with moviepy.editor
from moviepy import VideoFileClip, concatenate_videoclips
from torchvision import models, transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ==========================================
# PART 1: THE TRAINABLE BASE MODEL
# ==========================================

class MultimodalHighlightTransformer(nn.Module):
    """
    A Trainable Transformer architecture that fuses Visual, Audio, and Textual features
    to predict the 'importance score' of every second in a video.
    """

    def __init__(self, visual_dim=512, audio_dim=128, text_dim=768, d_model=512, nhead=8, num_layers=4):
        super(MultimodalHighlightTransformer, self).__init__()

        # 1. Feature Projection Layers (aligning different modalities to same dimension)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)

        # 2. Positional Encoding (to understand temporal sequence)
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model))  # Support up to 5000 segments

        # 3. Transformer Encoder (The "Brain")
        # Captures context: e.g., "A loud cheer (audio) usually follows a goal (visual)"
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4. Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Outputs probability 0.0 to 1.0
        )

    def forward(self, v_feat, a_feat, t_feat):
        """
        Input shapes: [Batch, Sequence_Length, Feature_Dim]
        """
        # Project features
        v = self.visual_proj(v_feat)
        a = self.audio_proj(a_feat)
        t = self.text_proj(t_feat)

        # Multimodal Fusion: We sum the embeddings (Late Fusion)
        # (Could also use concatenation, but summation preserves sequence length easier)
        x = v + a + t

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]

        # Pass through Transformer
        contextual_features = self.transformer(x)

        # Predict importance score for each time step
        scores = self.classifier(contextual_features)
        return scores


# ==========================================
# PART 2: THE INFERENCE PIPELINE
# ==========================================

def download_video(url, output_path=None):
    pass


class UniversalHighlightGenerator:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Generator on {self.device}...")

        # --- Load Feature Extractors ---

        # 1. Visual: ResNet18 (Pretrained on ImageNet)
        self.visual_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.visual_extractor.fc = nn.Identity()  # Remove classification layer
        self.visual_extractor.to(self.device).eval()
        self.visual_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Text: BERT (For semantic understanding of transcription)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased').to(self.device).eval()

        # 3. Audio: Whisper (For Transcription)
        print("Loading Whisper for Speech-to-Text...")
        self.asr_model = whisper.load_model("base", device=self.device)

        # 4. The Brain: Our Multimodal Transformer
        self.model = MultimodalHighlightTransformer().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loaded trained model weights.")
        else:
            print("WARNING: Using untrained model weights (Results will be random). Train the model first!")
        self.model.eval()

    def extract_features(self, video_path):
        """
        Breaks video into 1-second chunks and extracts V/A/T features.
        """
        print("Step 1: Extracting Features...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        # --- A. Visual Features (1 frame per second) ---
        visual_feats = []
        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_t = self.visual_transform(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.visual_extractor(frame_t)
                visual_feats.append(feat.cpu().numpy())
            else:
                visual_feats.append(np.zeros((1, 512)))  # Padding
        cap.release()

        # --- B. Audio Features (MFCCs) ---
        y, sr = librosa.load(video_path, sr=22050)
        audio_feats = []
        for sec in range(duration):
            start = sec * sr
            end = (sec + 1) * sr
            if end > len(y): break
            y_segment = y[start:end]
            mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=128)
            mfcc_mean = np.mean(mfcc, axis=1)  # Shape (128,)
            audio_feats.append(mfcc_mean)

        # --- C. Text Features (Whisper + BERT) ---
        print("Transcribing audio...")
        transcription = self.asr_model.transcribe(video_path)
        text_feats = [np.zeros(768) for _ in range(duration)]  # Initialize empty

        # Map transcription segments to seconds
        for segment in transcription['segments']:
            start = int(segment['start'])
            end = int(segment['end'])
            text = segment['text']

            # Embed text using BERT
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            # Use CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # Fill the features for the duration of the spoken sentence
            for i in range(start, min(end, duration)):
                text_feats[i] = emb

        # Pad or truncate to match lengths
        min_len = min(len(visual_feats), len(audio_feats), len(text_feats))
        return (np.array(visual_feats[:min_len]).squeeze(),
                np.array(audio_feats[:min_len]),
                np.array(text_feats[:min_len]))

    def generate_highlights(self, video_path, output_path, sensitivity_std=1.0):
        """
        Main pipeline.
        sensitivity_std: Controls dynamic length.
                         Lower (e.g., 0.5) = Longer highlights (More content included).
                         Higher (e.g., 1.5) = Shorter highlights (Only top moments).
        """
        # 1. Get Features
        v, a, t = self.extract_features(video_path)

        # 2. Convert to Tensor
        v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)  # Batch size 1
        a_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 3. Predict Importance
        print("Step 2: Predicting Segment Importance...")
        with torch.no_grad():
            # Output shape: [1, Seq_Len, 1]
            scores = self.model(v_tensor, a_tensor, t_tensor).squeeze().cpu().numpy()

        # 4. DYNAMIC LENGTH LOGIC (Statistical Thresholding)
        # Instead of fixed top-k, we calculate the statistical distribution of excitement
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score + (sensitivity_std * std_score)

        print(f"Dynamic Threshold Stats -> Mean: {mean_score:.4f}, Std: {std_score:.4f}, Threshold: {threshold:.4f}")

        # 5. Select Segments
        selected_indices = np.where(scores >= threshold)[0]

        if len(selected_indices) == 0:
            print("No significant highlights found. Lowering threshold...")
            threshold = mean_score  # Fallback to mean
            selected_indices = np.where(scores >= threshold)[0]

        # Group consecutive seconds into clips (e.g., if 5, 6, 7 are selected, make one clip 5-8)
        clips = []
        if len(selected_indices) > 0:
            start = selected_indices[0]
            prev = selected_indices[0]

            for i in selected_indices[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    clips.append((start, prev + 1))
                    start = i
                    prev = i
            clips.append((start, prev + 1))

        print(f"Identified {len(clips)} important segments.")

        # 6. Render Video
        print("Step 3: Rendering Final Video...")
        original_clip = VideoFileClip(video_path)
        final_clips = []

        # Add buffer (padding) to smooth cuts
        buffer = 1
        for s, e in clips:
            start_t = max(0, s - buffer)
            end_t = min(original_clip.duration, e + buffer)
            final_clips.append(original_clip.subclipped(start_t, end_t))

        if final_clips:
            final_video = concatenate_videoclips(final_clips)
            final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
            print(f"Success! Highlight saved to {output_path}")
            print(f"Original Length: {original_clip.duration:.2f}s -> Highlight Length: {final_video.duration:.2f}s")
        else:
            print("Error: No clips generated.")


# ==========================================
# PART 3: COMPLETE TRAINING LOOP
# ==========================================


class HighlightDataset(Dataset):
    """
    Dataset class for loading video features and highlight labels.
    Expects a directory structure:
    dataset_path/
        videos/
            video1.mp4
            video2.mp4
        labels/
            video1.json (contains per-second highlight scores)
            video2.json
    """

    def __init__(self, dataset_path, transform=None, cache_features=True):
        """
        Args:
            dataset_path: Path to dataset directory
            transform: Optional transform to apply
            cache_features: Whether to cache extracted features (saves time)
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.cache_features = cache_features
        self.feature_cache = {}

        # Find all video files
        self.video_files = glob.glob(os.path.join(dataset_path, "videos", "*.mp4"))
        self.video_files += glob.glob(os.path.join(dataset_path, "videos", "*.avi"))
        self.video_files += glob.glob(os.path.join(dataset_path, "videos", "*.mov"))

        print(f"Found {len(self.video_files)} videos in dataset")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]

        # Check if features are cached
        if video_path in self.feature_cache:
            return self.feature_cache[video_path]

        # Get corresponding label file
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        label_path = os.path.join(self.dataset_path, "labels", f"{video_name}.json")

        # Extract features (using your existing feature extraction)
        v_feat, a_feat, t_feat = self._extract_training_features(video_path)

        # Load labels
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            labels = torch.tensor(label_data['scores'], dtype=torch.float32)

        # Ensure lengths match
        min_len = min(v_feat.shape[0], a_feat.shape[0], t_feat.shape[0], labels.shape[0])
        v_feat = v_feat[:min_len]
        a_feat = a_feat[:min_len]
        t_feat = t_feat[:min_len]
        labels = labels[:min_len]

        # Convert to tensors
        v_tensor = torch.tensor(v_feat, dtype=torch.float32)
        a_tensor = torch.tensor(a_feat, dtype=torch.float32)
        t_tensor = torch.tensor(t_feat, dtype=torch.float32)

        sample = {
            'visual': v_tensor,
            'audio': a_tensor,
            'text': t_tensor,
            'labels': labels,
            'video_path': video_path
        }

        # Cache if requested
        if self.cache_features:
            self.feature_cache[video_path] = sample

        return sample

    def _extract_training_features(self, video_path):
        """
        Extract features for training (simplified version of your extract_features)
        You can reuse your existing UniversalHighlightGenerator's extract_features method
        """
        # This is a simplified version - you might want to use your existing
        # UniversalHighlightGenerator's extract_features method here

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        # Visual features (simplified - use your actual visual_extractor)
        visual_feats = []
        for sec in range(min(duration, 100)):  # Limit for demo
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if ret:
                # Use a simple feature (you should use your actual feature extractor)
                visual_feats.append(np.random.randn(512))  # Placeholder
            else:
                visual_feats.append(np.zeros(512))
        cap.release()

        # Audio features (simplified)
        audio_feats = [np.random.randn(128) for _ in range(len(visual_feats))]

        # Text features (simplified)
        text_feats = [np.random.randn(768) for _ in range(len(visual_feats))]

        return np.array(visual_feats), np.array(audio_feats), np.array(text_feats)


def create_sample_labels(video_path, output_json_path):
    """
    Helper function to create sample label files for training.
    In reality, you'd have human-annotated highlights.
    """
    cap = cv2.VideoCapture(video_path)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Create random labels for demo (0 or 1)
    # In real use, these would be human-annotated
    scores = np.random.choice([0, 1], size=duration, p=[0.7, 0.3])

    label_data = {
        'video': os.path.basename(video_path),
        'duration': duration,
        'scores': scores.tolist()
    }

    with open(output_json_path, 'w') as f:
        json.dump(label_data, f, indent=2)


def train_model(dataset_path, epochs=10, batch_size=4, learning_rate=1e-4,
                save_path="highlight_model.pth", device='cuda'):
    """
    Complete training function for the MultimodalHighlightTransformer.

    Args:
        dataset_path: Path to dataset directory (with videos/ and labels/ subfolders)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Where to save the trained model
        device: Device to train on ('cuda' or 'cpu')
    """

    print("=" * 60)
    print("TRAINING MULTIMODAL HIGHLIGHT TRANSFORMER")
    print("=" * 60)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    print("\n1. Loading dataset...")
    dataset = HighlightDataset(dataset_path, cache_features=True)

    if len(dataset) == 0:
        print("Error: No videos found in dataset!")
        return

    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=custom_collate)

    print(f"   Train samples: {train_size}, Val samples: {val_size}")

    # Initialize model
    print("\n2. Initializing model...")
    model = MultimodalHighlightTransformer().to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\n3. Starting training...")
    print("-" * 60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            v_feat = batch['visual'].to(device)
            a_feat = batch['audio'].to(device)
            t_feat = batch['text'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(v_feat, a_feat, t_feat)

            # Remove extra dimension
            outputs = outputs.squeeze(-1)

            # Calculate loss (need to handle variable sequence lengths)
            loss = 0
            for i in range(outputs.shape[0]):  # For each video in batch
                # Get valid length (non-padded part)
                valid_len = batch['lengths'][i]
                loss += criterion(
                    outputs[i, :valid_len],
                    labels[i, :valid_len]
                )
            loss = loss / outputs.shape[0]  # Average over batch

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

            # Print progress
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                v_feat = batch['visual'].to(device)
                a_feat = batch['audio'].to(device)
                t_feat = batch['text'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(v_feat, a_feat, t_feat)
                outputs = outputs.squeeze(-1)

                val_loss = 0
                for i in range(outputs.shape[0]):
                    valid_len = batch['lengths'][i]
                    val_loss += criterion(
                        outputs[i, :valid_len],
                        labels[i, :valid_len]
                    )
                val_loss = val_loss / outputs.shape[0]

                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"   ✓ New best model saved! (Loss: {avg_val_loss:.4f})")

        # Print epoch summary
        print(f"\n   Epoch {epoch + 1} Summary:")
        print(f"     Train Loss: {avg_train_loss:.4f}")
        print(f"     Val Loss: {avg_val_loss:.4f}")
        print(f"     Best Val Loss: {best_val_loss:.4f}")
        print("-" * 60)

    print("\n4. Training complete!")
    print(f"   Best model saved to: {save_path}")
    print(f"   Final Train Loss: {train_losses[-1]:.4f}")
    print(f"   Final Val Loss: {val_losses[-1]:.4f}")

    # Plot training curves (optional)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png')
        print("   Training curve saved to 'training_curve.png'")
    except ImportError:
        pass

    return model


def custom_collate(batch):
    """
    Custom collate function to handle variable sequence lengths.
    Pads sequences to the same length within a batch.
    """
    # Find max length in batch
    max_len = max(item['visual'].shape[0] for item in batch)
    feat_dim_v = batch[0]['visual'].shape[1]
    feat_dim_a = batch[0]['audio'].shape[1]
    feat_dim_t = batch[0]['text'].shape[1]

    # Initialize padded tensors
    visual_batch = torch.zeros(len(batch), max_len, feat_dim_v)
    audio_batch = torch.zeros(len(batch), max_len, feat_dim_a)
    text_batch = torch.zeros(len(batch), max_len, feat_dim_t)
    labels_batch = torch.zeros(len(batch), max_len)
    lengths = []

    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item['visual'].shape[0]
        lengths.append(seq_len)
        visual_batch[i, :seq_len] = item['visual']
        audio_batch[i, :seq_len] = item['audio']
        text_batch[i, :seq_len] = item['text']
        labels_batch[i, :seq_len] = item['labels']

    return {
        'visual': visual_batch,
        'audio': audio_batch,
        'text': text_batch,
        'labels': labels_batch,
        'lengths': lengths
    }


    # print("\nTo train the model:")
    # print("1. Prepare your dataset with the structure:")
    # print("   dataset/")
    # print("   ├── videos/")
    # print("   │   ├── video1.mp4")
    # print("   │   └── video2.mp4")
    # print("   └── labels/")
    # print("       ├── video1.json")
    # print("       └── video2.json")
    # print("\n2. Run:")
    # print("   train_model('path/to/dataset', epochs=20)")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    # Choose which mode to run
    mode = input("Enter 'train' to train model or 'run' to generate highlights: ").strip()

    if mode.lower() == 'train':
        # Train the model
        dataset_path = input("Enter path to your dataset: ").strip()
        train_model(
            dataset_path=dataset_path,
            epochs=20,
            batch_size=4,
            save_path="highlight_model.pth"
        )
    else:

        # 1. Initialize Generator

        generator = UniversalHighlightGenerator(model_path="highlight_model.pth")

        # 2. Process a Video

        video_source = input("Enter video file path OR video link: ").strip()

        try:
            if video_source.startswith(("http://", "https://", "www.", "youtu.be", "youtube.com")):
                print("Detected as online video link")
                input_video = download_video(video_source)
            else:
                print("Detected as local file path")
                input_video = video_source
            output_video = "output_highlights.mp4"

            if os.path.exists(input_video):
                print(f"Processing video: {input_video}")
                generator.generate_highlights(input_video, output_video, sensitivity_std=0.8)
            else:
                print(f"Error: Video file not found at {input_video}")

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            import traceback

            traceback.print_exc()

