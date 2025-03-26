!rm -rf prostate-cancer-thesis
!git clone "https://github.com/HyphenSaad/prostate-cancer-thesis.git"

!mkdir prostate-cancer-thesis/encoders/ckpts

ACCESS_TOKEN = ""

!echo -e "\nDownloading CTransPath Encoder..."
UNI2_FID = "1GjcTdgj8xU_w3USz9rj3hiwbAI41YmpG"
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{UNI2_FID}?alt=media -o ctranspath.pth
!mv ctranspath.pth prostate-cancer-thesis/encoders/ckpts/ctranspath.pth

!echo -e "\nInstalling Modified Timm Library..."
TIMM_FID = "1--He4NUKI341_5ubsKoKHZZq3RSliyds"
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{TIMM_FID}?alt=media -o timm-0.5.4.tar
!pip install timm-0.5.4.tar

!echo -e "\nCleaning Up..."
!rm -rf timm-0.5.4.tar