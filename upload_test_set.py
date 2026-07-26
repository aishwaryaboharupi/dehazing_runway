import os
from huggingface_hub import HfApi, login

# =========================================================
# CONFIGURATION
# =========================================================
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
REPO_ID = "NeuroPropel/CockpitAI_dehaze_dataset"

LOCAL_TEST_HAZY = r"C:\Users\ACER\Desktop\Cockpit_AI\LARD_dataset\test\hazy"
LOCAL_TEST_CLEAN = r"C:\Users\ACER\Desktop\Cockpit_AI\LARD_dataset\test\clean"

print("Logging into Hugging Face...")
if HF_TOKEN != "YOUR_HF_TOKEN_HERE":
    login(token=HF_TOKEN)

api = HfApi()

print("Uploading test set to Hugging Face...")

if os.path.exists(LOCAL_TEST_HAZY):
    api.upload_folder(
        folder_path=LOCAL_TEST_HAZY,
        path_in_repo="test_data/hazy",
        repo_id=REPO_ID,
        repo_type="dataset"
    )

if os.path.exists(LOCAL_TEST_CLEAN):
    api.upload_folder(
        folder_path=LOCAL_TEST_CLEAN,
        path_in_repo="test_data/clean",
        repo_id=REPO_ID,
        repo_type="dataset"
    )

print("Test dataset uploaded successfully!")