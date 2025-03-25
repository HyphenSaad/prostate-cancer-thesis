BATCH_NUMBER = 0

S1_CREATE_PATCHES = ['15PpetOtzwZ6c2O4LMHjVL5hBeOVhhupn', '15UYAISLa_Sil8aEF1SNP1s3ZNDmcRhU3', '16UR8OQOTvJuEbP3mbj8h7k599AsYAmeQ', '17tTtyH7te8E890HNh3ozFCEOT-VsuRnQ', '17RfT7hxwix4fmyrxMq-xKxBAwj7R6Rtk', '17N0mPpBDki4N-ldQmfJhr5MGc8y0HyP6', '177H9sRgBYhQYPY-cWFjOKFy6mdSC5muH', '176aqMhG2J-YG7MV3vFOqmPywem21kLuC', '16w3Xgph0LmBJqVGmmSnFADkF8nkJeyp-', '16uN75FryEs7ZDoV04vuE4Q2IuE6PdTz1', '167Nt5mQSU04C-r2QZdK621MqETxcekaZ', '166DovyohN5eKgNz-m3K1SKHHptRWzMVs', '15zSU_zUPh8bTEe3rt-Qsz_VjnfgzmNua', '15qbj5aHd9dAbNDwR41TG8j3dNrXstytA', '15oyN0KvWEGxYtVANjWFEQHkCYb-HVT2D', '15oUPXsbHVIu-HZ83S1UuH2sF9Gg5S0yC', '15nv_qpJFOsegfqXC2xj9VLgP6ieSZpUa', '15ibvoPPUXXiHA0WZigaazMxa6U9YG3mv', '15bNFw79ZzeeRQK7mj-6ayOfSwdmAwQAG', '15am4N8f1hVgxX0g0hV4fk23f5Et9XlIL', '15U7OUn1joaNYavzUUofKhqscVXbqiiPy', '15SDMViOcMVZmT25PSh1WKs6a6IDEoNzS', '15S9W1ESSiQiHkg0tmCSycniNhDHTMlff', '15RDBHtLP0MUQY_gDUQU1l_b1aMrb0ED-', '16t5KyS-HLkNOLp7aDIydKGXsmcV1DWj6', '16qJfbkixTZk_9qoBbjgEg0oFupOD7Hz7', '16iTdaDVlcNfhUejO78VyNZ50-OVtQ-AY', '16ZyiJFTgQ82fVqYLaj8jqjRr3wn5798b', '16YFQBfxCHWUcnuW-4zG1EtL14vMaXW2s', '16V-98LzOeEzkobevhvZKUyeIajweWyb_', '16RiaP6L_xGEulRfUGosy9Bo6Uvj2ac1S', '16Lik9tFCvIjWpBA_Y7UEmoI1h_puRZFy', '16LhsbjGlxZZv5fTJN4qXa73UcvsEIb92', '16Kpq59qw4Yy29Nkzyd9ZJgPShdiB6m3G', '16II-Ao3El4B6zrwc4RAmPk375D1QnDKw', '16CTsixWdWrHJ3k2MXDO1nzTYuvxobhBH', '16B-hqgtJufTpVrswir3cUEpZttm-8Ogk', '168O_c85Zm0oAuZnPvtFJWjyPBJXxvKa5', '17xNLaxDk2Ei-8K16ZWPb44sCiG2XJVWL', '17ufKcRs1FBMpueZS2WY-XmCOU4_8JppU', '17acxYO_uQzhLHdER5eRSunJoXdAL5-jv', '17UVmcgYBMIptFz5v0eVh8CNoDZX5p_8v', '17SehmlpMZxmWeAfkA65x3-Jt3-fQqGF4']

FID = S1_CREATE_PATCHES[BATCH_NUMBER]

ACCESS_TOKEN = ""

!echo -e "Downloading Create Patches Files..."
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{FID}?alt=media -o output_{BATCH_NUMBER}_create_patches.zip

!echo -e "\nUnzipping Create Patches Files..."
!mkdir output_{BATCH_NUMBER}
!unzip -q output_{BATCH_NUMBER}_create_patches.zip -d output_{BATCH_NUMBER}

!echo -e "\nDeleting Extra Files..."
!rm -rf output_{BATCH_NUMBER}_create_patches.zip

!python "prostate-cancer-thesis/scripts/extract_patches.py" --patch-size 512 --output-base-directory "output_{BATCH_NUMBER}"

!rm -rf output_{BATCH_NUMBER}/create_patches

!echo "Creating ZIP File For Batch # {BATCH_NUMBER}"
!cd output_{BATCH_NUMBER}; zip -qr -1 "output_{BATCH_NUMBER}.zip" . ; mv "output_{BATCH_NUMBER}.zip" .. ; echo "ZIP File Created For Batch # {BATCH_NUMBER}"

!echo "Removing Directory For Batch # {BATCH_NUMBER}"
!rm -r "output_{BATCH_NUMBER}"; echo "Directory Removed For Batch # {BATCH_NUMBER}"