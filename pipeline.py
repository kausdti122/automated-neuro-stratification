import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, EnsureType

# ==========================================
# 0. GENERATE SYNTHETIC NIfTI DATA FOR TESTING
# ==========================================
print("--- Step 0: Generating Mock Clinical Imaging Data ---")
os.makedirs("sample_data", exist_ok=True)

# Generate 30 mock patient scans to simulate a small cohort
np.random.seed(42)
for i in range(30):
    # Create mock 3D MRI brain scan (64x64x64 voxel grid)
    mock_mri = np.random.rand(64, 64, 64)
    # Add a fake structural "brain sphere" in the center to simulate tissue density
    x, y, z = np.indices((64, 64, 64))
    mask = (x-32)**2 + (y-32)**2 + (z-32)**2 < 20**2
    mock_mri[mask] += 0.5
    
    # Save as NIfTI file using a standard diagonal identity affine matrix
    ni_img = nib.Nifti1Image(mock_mri, affine=np.eye(4))
    nib.save(ni_img, f"sample_data/sub-{i:02d}_T1w.nii.gz")

print("Successfully generated 30 synthetic NIfTI volumes in './sample_data/'\n")

# ==========================================
# PHASE 1: DATA INGESTION & PREPROCESSING
# ==========================================
print("--- Phase 1: Ingesting & Normalizing NIfTI Volumes ---")

def ingest_and_normalize(file_path):
    # Load NIfTI file using NiBabel
    mri_loaded = nib.load(file_path)
    # Extract coordinate affine matrix for spatial tracking
    affine = mri_loaded.affine
    # Extract structural volume as a NumPy array
    volume_array = mri_loaded.get_fdata()
    # Min-Max Voxel Intensity Normalization
    normalized_volume = (volume_array - np.min(volume_array)) / (np.max(volume_array) - np.min(volume_array))
    return normalized_volume, affine

# Test on the first subject
sample_vol, sample_affine = ingest_and_normalize("sample_data/sub-00_T1w.nii.gz")
print(f"Loaded Volume Shape: {sample_vol.shape}")
print(f"Extracted Spatial Affine Matrix:\n{sample_affine}\n")

# ==========================================
# PHASE 2: VOLUMETRIC DEEP LEARNING SEGMENTATION
# ==========================================
print("--- Phase 2: Deploying MONAI 3D UNet Architecture ---")

# Setup torch device acceleration (Utilizing Colab's Free T4 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate a MONAI 3D UNet 
# Configured for 1 input channel (MRI intensity) and 1 output channel (Region Mask)
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64),
    strides=(2, 2),
).to(device)
model.eval() # Set model to evaluation mode

# Preprocessing transforms pipeline utilizing updated MONAI 1.3+ configurations
transforms = Compose([
    EnsureChannelFirst(channel_dim='no_channel'), # Safely sets structural channel dimensions
    ScaleIntensity(),
    EnsureType()
])

# Process the sample volume through our network
input_tensor = transforms(sample_vol).unsqueeze(0).to(device) # Add batch dimension
with torch.no_grad():
    segmentation_output = model(input_tensor)
    # Apply a sigmoid threshold to create a binary mask array
    binary_mask = (torch.sigmoid(segmentation_output) > 0.5).squeeze().cpu().numpy()

print(f"Generated 3D Segmentation Mask Shape: {binary_mask.shape}\n")

# ==========================================
# PHASE 3: FEATURE ENGINEERING & COHORT STRATIFICATION
# ==========================================
print("--- Phase 3: Feature Engineering and Patient Clustering ---")

# Simulate downstream feature metrics for our 30 subjects
cohort_data = []
total_intracranial_volume = 262144 # 64x64x64 total voxels

for i in range(30):
    # Simulate slightly varying segmented cortical volumes
    segmented_volume = np.random.randint(15000, 35000)
    # Calculate engineered feature: Atrophy Index
    atrophy_index = segmented_volume / total_intracranial_volume
    
    # Simulate correlated Tau-PET tracer accumulation scores (higher atrophy = higher tau)
    tau_pet_score = (1.5 - atrophy_index * 2) + np.random.normal(0, 0.1)
    
    cohort_data.append({
        "Subject_ID": f"sub-{i:02d}",
        "Cortical_Volume": segmented_volume,
        "Atrophy_Index": atrophy_index,
        "Tau_PET_Score": tau_pet_score
    })

df = pd.DataFrame(cohort_data)

# Deploy Unsupervised Learning via Scikit-Learn KMeans
# Stratifying patients into 3 cohorts: Mild/Stable, Lateralized PPA, or Severe pathology
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cohort_Label"] = kmeans.fit_predict(df[["Atrophy_Index", "Tau_PET_Score"]])

# Map cluster integers to clinical descriptive strings
cluster_mapping = {0: "Mild / Stable Atrophy", 1: "Severe Multi-focal Pathology", 2: "Tau-Dominant PPA Model"}
df["Clinical_Cohort"] = df["Cohort_Label"].map(cluster_mapping)

print(df[["Subject_ID", "Atrophy_Index", "Tau_PET_Score", "Clinical_Cohort"]].head())
print("\n--- Pipeline Execution Complete! Generating Plots... ---")

# ==========================================
# PHASE 4: COHORT VISUALIZATION
# ==========================================
# Create a multi-panel figure formatted for scientific publishing (300 DPI layout)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: 2D Mid-Sagittal Cutout Slice Overlay of Segmentation
mid_slice = 32
ax1.imshow(sample_vol[:, :, mid_slice], cmap="gray")
ax1.imshow(binary_mask[:, :, mid_slice], cmap="jet", alpha=0.4) # Transparent mask overlay
ax1.set_title("Panel A: 3D MONAI Segmentation Mask Overlay (Mid-Slice)", fontsize=11, fontweight='bold')
ax1.axis("off")

# Panel B: Scatter Plot of Patient Stratification Cohorts
colors = {0: '#2ca02c', 1: '#d62728', 2: '#1f77b4'}
labels = {0: 'Mild / Stable', 1: 'Severe Multi-focal', 2: 'Tau-Dominant PPA'}

for g in np.unique(df.Cohort_Label):
    ix = np.where(df.Cohort_Label == g)
    ax2.scatter(df.Atrophy_Index.iloc[ix], df.Tau_PET_Score.iloc[ix], 
                c=colors[g], label=labels[g], s=70, edgecolor='black', alpha=0.8)

ax2.set_title("Panel B: Multi-Modal Patient Cohort Stratification", fontsize=11, fontweight='bold')
ax2.set_xlabel("Engineered Atrophy Index (Cortical / Total Volume)", fontsize=10)
ax2.set_ylabel("Standardized Tau-PET Tracer Uptake Score", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(title="Stratified Cohorts", loc="upper right")

plt.tight_layout()
# Save the figure instantly as a high-resolution PNG
plt.savefig("pipeline_results_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n Figure saved successfully as 'pipeline_results_300dpi.png' at 300 DPI.")
