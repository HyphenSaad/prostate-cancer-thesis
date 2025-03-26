!rm -rf prostate-cancer-thesis
!git clone "https://github.com/HyphenSaad/prostate-cancer-thesis.git"

!mkdir prostate-cancer-thesis/encoders/ckpts

ACCESS_TOKEN = ""

!echo -e "\nDownloading UNI2 Encoder..."
UNI2_FID = "1I5xIYcIGrH_1GLY8gK2C0e0jKo0k8sYm"
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{UNI2_FID}?alt=media -o uni2.bin
!mv uni2.bin prostate-cancer-thesis/encoders/ckpts/uni2.bin

!echo -e "\nLogging In To Hugging Face..."
!git config --global credential.helper store
!huggingface-cli login --token "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" --add-to-git-credential