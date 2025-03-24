BATCH_NUMBER = 0

!git clone "https://github.com/HyphenSaad/prostate-cancer-thesis.git"

S4_CREATE_PATCHES = ['15dWuU21YB3qE5HeRw_CORrRPRZQUiJNf', '166FubrpZQJxIVOHUhgHlUUbfSfwFtDGR', '17-3vwOD0cAY2cBAi0S4Z4S7uCfV3YU4E', '17iKFDqKdfYutU8Zz3RdUiVroUFhVaWcs', '17mVeR_ehVF8HAaiyDbpyFueqQWVp-OGo', '17w0NBsEW21kgXPj8vYYjEO3PkMIrx3S3', '17xmrb1bLlZT2-S1DSe1xi3BZw-9AETDc', '186WJLHnWHgux7UgKRpbN798cTINHo1cA', '188QZyPKYpUuQa5skTwRZNl3QV6Wk5nov', '18CH4WPM9QaQ9QK1rnxD-y9RIcQwMo9lx', '15flTxg4x13JKtfQx-tSwW-CJVOuv7_v-', '15hzPg4MZE2pom8oTPgg61ZmgvAgWBcVL', '15kFkztdVcufT4Fx-X2RiJPUu6Yd7t-tD', '15qLfiYQB_-1wBeKsHL8MOp1ZmSQsCyXG', '15qypZqzU8py-858-NyrLSLhvXzxemVYJ', '15w1q5Nbbze6I1ifSx-wdIGx3302PhNET', '15yPdQqKfN9QgE2RqER1FSMAWR7pGmFj4', '161pe7BEPo8eQBptuqJllNcbp6KhpUOwI', '164eIJHKE2iFzlv6vENeZiuY7JQRi-xcz', '165wy_ZknjZRLRPPTVepAXaU6LmLlNQPS', '16Akw8TGvIGYfSYvt6FvAGTEipCyFOk2w', '16BBwD6L5B626PwnJupWaZ8BGTJS-IGdN', '16OFS7LNQQLnqTXCLmFXj6Ee3puL3iP2C', '16PtRYxuifHzm8xkt2QsrWMu3DRyokYoQ', '16Vsjn6VFrEs-1GWe7Bp-Uss59xY217ms', '16XjYWgeepDY79gvplA0gAhbjcDDDLW7u', '16d6dMJjTFJA15npENQKBZ8-INlc42ZjG', '16mXPkkqmLsdM1D1T1GZuB-jxXe7X1lA1', '16rk9lTamYFF_6yFAqYA39o9vxWn1dvFl', '16wehRrysPkvpJs4RJwzL-d0EBeMJ8vYP', '179SYftw94hsN2A-7SbntMiEjR3_peEo8', '17AuiFlt2gnGjpXSKpNZckTk3Vn1m19Wt', '17JH6pJFDmQwS7YRVmYzvB98JlChiHtNr', '17L4Pfu-igUEn5zxWTfibLJEHEp_OjoKl', '17OAu-SmdZAa3CNUqAniG3dPZk-lyE-Yj', '17OMIHRGu2B-m6_1A37KWDFRdnNV-2R9F', '17V-WBhBGVbcUfKLc5z-LlRFWIYTuPEXR', '17XtzfYNN3OvFN6t9PNLl_9B6hfH5lnCV', '17ehhUL695SXBEO2TFt3METXTKj0-BxWF', '17hoQPt4DbkH4ZLDx7Gk77eP3UTf2yo-2', '17iYwDLxKUEr_vbt2UnufdwketSxbstpg', '17k_RV1WV_VlACEqxlgH6eRAvriaPU69o', '17t-hX7AeL9dI9PFPecns2TZnHdgw-Qg2']

FID = S4_CREATE_PATCHES[BATCH_NUMBER]

ACCESS_TOKEN = ""

!echo -e "Downloading Create Patches Files..."
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{FID}?alt=media -o output_{BATCH_NUMBER}_create_patches.zip

!echo -e "\nUnzipping Create Patches Files..."
!mkdir output_{BATCH_NUMBER}
!unzip -q output_{BATCH_NUMBER}_create_patches.zip -d output_{BATCH_NUMBER}

!echo -e "\nDeleting Extra Files..."
!rm -rf output_{BATCH_NUMBER}_create_patches.zip

!python "prostate-cancer-thesis/scripts/extract_patches.py" --patch-size 256 --output-base-directory "output_{BATCH_NUMBER}"

!rm -rf output_{BATCH_NUMBER}/create_patches

!echo "Creating ZIP File For Batch # {BATCH_NUMBER}"
!cd output_{BATCH_NUMBER}; zip -qr -1 "output_{BATCH_NUMBER}.zip" . ; mv "output_{BATCH_NUMBER}.zip" .. ; echo "ZIP File Created For Batch # {BATCH_NUMBER}"

!echo "Removing Directory For Batch # {BATCH_NUMBER}"
!rm -r "output_{BATCH_NUMBER}"; echo "Directory Removed For Batch # {BATCH_NUMBER}"