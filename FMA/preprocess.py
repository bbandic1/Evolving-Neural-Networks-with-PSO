import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

METADATA_PATH = 'data/raw/fma_metadata/tracks.csv'
AUDIO_ROOT_PATH = 'data/raw/fma_small'
OUTPUT_PATH = 'data/processed/spectrograms'

SAMPLE_RATE = 22050  
DURATION = 30        
N_MELS = 128         
HOP_LENGTH = 512     
N_FFT = 2048         

def get_audio_path(audio_dir, track_id):
    tid_str = f'{track_id:06d}'
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def main():
    print("Starting audio preprocessing...")
    print(f"Loading metadata from {METADATA_PATH}...")
    tracks_df = pd.read_csv(METADATA_PATH, header=[0, 1], index_col=0)
    
    small_subset_tracks = tracks_df[tracks_df[('set', 'subset')] == 'small']
    
    print(f"Found {len(small_subset_tracks)} tracks in the 'small' subset.")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Output will be saved to {OUTPUT_PATH}")

    for track_id, row in tqdm(small_subset_tracks.iterrows(), total=len(small_subset_tracks), desc="Processing tracks"):
        
        track_id = int(track_id)
        output_file_path = os.path.join(OUTPUT_PATH, f'{track_id:06d}.npy')
        if os.path.exists(output_file_path):
            continue

        audio_path = get_audio_path(AUDIO_ROOT_PATH, track_id)

        if not os.path.exists(audio_path):
            continue
            
        try:

            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
            
            target_length = DURATION * sr
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            np.save(output_file_path, log_mel_spec)

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue

    print("\nPreprocessing finished successfully!")
    print(f"{len(os.listdir(OUTPUT_PATH))} files were created in {OUTPUT_PATH}")

if __name__ == '__main__':
    main()