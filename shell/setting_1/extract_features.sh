BATCH_NUMBER = 0

!rm -rf output_{BATCH_NUMBER}.zip

S1_CREATE_PATCHES = ['15PpetOtzwZ6c2O4LMHjVL5hBeOVhhupn', '15UYAISLa_Sil8aEF1SNP1s3ZNDmcRhU3', '16UR8OQOTvJuEbP3mbj8h7k599AsYAmeQ', '17tTtyH7te8E890HNh3ozFCEOT-VsuRnQ', '17RfT7hxwix4fmyrxMq-xKxBAwj7R6Rtk', '17N0mPpBDki4N-ldQmfJhr5MGc8y0HyP6', '177H9sRgBYhQYPY-cWFjOKFy6mdSC5muH', '176aqMhG2J-YG7MV3vFOqmPywem21kLuC', '16w3Xgph0LmBJqVGmmSnFADkF8nkJeyp-', '16uN75FryEs7ZDoV04vuE4Q2IuE6PdTz1', '167Nt5mQSU04C-r2QZdK621MqETxcekaZ', '166DovyohN5eKgNz-m3K1SKHHptRWzMVs', '15zSU_zUPh8bTEe3rt-Qsz_VjnfgzmNua', '15qbj5aHd9dAbNDwR41TG8j3dNrXstytA', '15oyN0KvWEGxYtVANjWFEQHkCYb-HVT2D', '15oUPXsbHVIu-HZ83S1UuH2sF9Gg5S0yC', '15nv_qpJFOsegfqXC2xj9VLgP6ieSZpUa', '15ibvoPPUXXiHA0WZigaazMxa6U9YG3mv', '15bNFw79ZzeeRQK7mj-6ayOfSwdmAwQAG', '15am4N8f1hVgxX0g0hV4fk23f5Et9XlIL', '15U7OUn1joaNYavzUUofKhqscVXbqiiPy', '15SDMViOcMVZmT25PSh1WKs6a6IDEoNzS', '15S9W1ESSiQiHkg0tmCSycniNhDHTMlff', '15RDBHtLP0MUQY_gDUQU1l_b1aMrb0ED-', '16t5KyS-HLkNOLp7aDIydKGXsmcV1DWj6', '16qJfbkixTZk_9qoBbjgEg0oFupOD7Hz7', '16iTdaDVlcNfhUejO78VyNZ50-OVtQ-AY', '16ZyiJFTgQ82fVqYLaj8jqjRr3wn5798b', '16YFQBfxCHWUcnuW-4zG1EtL14vMaXW2s', '16V-98LzOeEzkobevhvZKUyeIajweWyb_', '16RiaP6L_xGEulRfUGosy9Bo6Uvj2ac1S', '16Lik9tFCvIjWpBA_Y7UEmoI1h_puRZFy', '16LhsbjGlxZZv5fTJN4qXa73UcvsEIb92', '16Kpq59qw4Yy29Nkzyd9ZJgPShdiB6m3G', '16II-Ao3El4B6zrwc4RAmPk375D1QnDKw', '16CTsixWdWrHJ3k2MXDO1nzTYuvxobhBH', '16B-hqgtJufTpVrswir3cUEpZttm-8Ogk', '168O_c85Zm0oAuZnPvtFJWjyPBJXxvKa5', '17xNLaxDk2Ei-8K16ZWPb44sCiG2XJVWL', '17ufKcRs1FBMpueZS2WY-XmCOU4_8JppU', '17acxYO_uQzhLHdER5eRSunJoXdAL5-jv', '17UVmcgYBMIptFz5v0eVh8CNoDZX5p_8v', '17SehmlpMZxmWeAfkA65x3-Jt3-fQqGF4']
S1_EXTRACT_PATCHES = ['17zOzYy4C2IwmbcE_eYJgOOldH4SLp2nd', '188YGofDu41-AWMiHc0EbUH9VNHEycEfY', '18DD_zNnGXNZwP_f_0e-kRZFQ-_trFKf3', '18DGnNOFyu2ABCMiZzk74vGwpZkRiBHKN', '18FblfJrgARmTxcY-1WqfMHtfIkijuTCB', '18MVECeXtXcntInwjXqKrDAKwwHs2SFcR', '18OQx9gz5gzf_TLh269HUbUJzOUwAjp-C', '18VXsrO2QQDpofFezdo9EpmUKjm68adJ2', '18o67RsbONWolo0uRR0fR9OKuYu5Xn9i7', '18pwNPSoERJBINjIe2iAlGMI-32QU3k3Y', '18tCXXg4vXA2pAQb8-hWtHOmlT_dynocB', '18wUIpiuTjzDPV33FTjrhbQXIgGRqZiL2', '19215yDz4eyNur7r1Fz5ElQcji2a2WOhC', '1935NV2FzPeudtYB-fSFIwEQKssc6nGTg', '1957w73Wxx-pQhSLyV9JwP0YvplqOn5HL', '19MHVMf-7XNFsLuL0SYH2Wc1fd1XsmOhD', '19MyP2YZvRFEcCFPBUvPKjz1sqBy1YfjS', '19PO4Z5zsi3c5zui1Ya2y8pP8HQntSj7e', '19RXxakAhfhgtsK9ZPJ7jrglCDXKc33qJ', '19YkIC60708-EoC-kXlYaxJ24IbFteRBc', '19cl2t0unna2rYlycrzt15CLup8AYHOZj', '19lNQxblXuwPr7gmOij55eXoO-iPzMUBQ', '19n6DuecTjARyeekSmXmbPvIdSJdowPC5', '19qz3_poHWjxS_HdVg-QtcsmVu_HUz5xM', '19xG5KGi2Zxy0Fwdy58vRM7orgNpyjtdA', '1A1H-_5PZbJKeVXD_73ajNxVm-UIqrq6l', '1A5F4DIrlRrGUSFWk12bMG8junUBERTSp', '1A9hBUgjNg9yh8NcRJo2N0S615NbtIpmW', '1ADGNb4jlsRwI0syksOw-ZOk0D3TTjaot', '1ALG5Zc7doaewr_4zoZmmS802wnkQT3Ur', '1AMBl-2fmgKFC4uPdfJInFteSS9Oar-vn', '1AWlZiiQzPPduEKerkoFtiO_5Py6LAjgW', '1AepsIE586g8nupAtcEE0_CfObFEuEpTr', '1AhtprCLCKuJr_31yZoGCl7u1BQ0N9yfe', '1AmEDxC8L-suGFKl5V_IcdWjsDn3RdqNb', '1AronEAvxbfqyh_OBRuT-WjFyU7Nj5FRy', '1AtOUOqJ9to_cJCfudlJfS0LgsMMfM73_', '1Au2drOr2a-mtChg2n3tC9IvyT_x0IR9Q', '1Aw_0MB5Tr8b2He1a3wS8XohLIO0j8isA', '1AxM5GV9osZ1qyf3DT_26sKJ-VcdQv9f2', '1AxbutB9yE7ti_jpD1W4CmLb9_oDbEwDI', '1AyB8Sajwjm1QnfYwXXB94FisSQaHrjK9', '1AyRZQ1qTTnrB7wz7e1cUZURioH7qcpQ0']

CREATE_PATCH_FID = S1_CREATE_PATCHES[BATCH_NUMBER]
EXTRACT_PATCH_FID = S1_EXTRACT_PATCHES[BATCH_NUMBER]

ACCESS_TOKEN = ""

!echo -e "Downloading Create Patches Files..."
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{CREATE_PATCH_FID}?alt=media -o output_{BATCH_NUMBER}_create_patches.zip

!echo -e "\nUnzipping Create Patches Files..."
!mkdir output_{BATCH_NUMBER}
!unzip -q output_{BATCH_NUMBER}_create_patches.zip -d output_{BATCH_NUMBER}

!echo -e "Downloading Extract Patches Files..."
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{EXTRACT_PATCH_FID}?alt=media -o output_{BATCH_NUMBER}_extract_patches.zip

!echo -e "\nUnzipping Extract Patches Files..."
!unzip -q output_{BATCH_NUMBER}_extract_patches.zip -d output_{BATCH_NUMBER}

!echo -e "\nDeleting Extra Files..."
!rm -rf output_{BATCH_NUMBER}_create_patches.zip
!rm -rf output_{BATCH_NUMBER}_extract_patches.zip

!python "prostate-cancer-thesis/scripts/extract_features.py" --encoder "uni2" --output-base-directory "output_{BATCH_NUMBER}" --dataset-base-directory "prostate-cancer-thesis" --dataset-info-file-name "train.csv" --patch-size 512

!rm -rf output_{BATCH_NUMBER}/create_patches
!rm -rf output_{BATCH_NUMBER}/extract_patches

!echo "Creating ZIP File For Batch # {BATCH_NUMBER}"
!cd output_{BATCH_NUMBER}; zip -qr -1 "output_{BATCH_NUMBER}.zip" . ; mv "output_{BATCH_NUMBER}.zip" .. ; echo "ZIP File Created For Batch # {BATCH_NUMBER}"

!echo "Removing Directory For Batch # {BATCH_NUMBER}"
!rm -r "output_{BATCH_NUMBER}"; echo "Directory Removed For Batch # {BATCH_NUMBER}"