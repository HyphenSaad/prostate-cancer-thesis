BATCH_NUMBER = 0

ACCESS_TOKEN = ""
ENCODER = "ctranspath"

!rm -rf output_{BATCH_NUMBER}.zip

S2_CREATE_PATCHES = ['1-L91eWihyI4Tsow1rRtfendzD8pJGwfg', '1-MI26s6e00bQ1sfVgvVyU4KGrRHvucxo', '1-N4I8hhiZq_sYQi0xL3qM2vxcP7gGqWL', '1-OFN5S_e1rpb6sqo1JIjX6rsewq2IOm_', '1-PrMgOGfLiw8LZ0hISRd9fZwQZizIAHW', '1-QBffAnNpcbRw6AWebBpiInvtMBSPBVT', '1-QPTvMF0wMVSYI3SdfOVm5Wnz9f2VxHw', '1-R8HKZDJBT7eqhryAM14be7nqNkoURON', '1-RZLJjyjfw8rzlstTUfn2yz8hUz84DpF', '1-L8fCvS4QW2VLw1sXFYH7uBpv2gLF9AR', '1-CA0bhg2vHJHnTtZtEmk6t_BmVQ_C5xN', '1-DM8YROZ_uQE6cMvDIlQC2m-JQay-fkr', '1-E9j1vMFYjmIqYcWD-PguSpLtzjfgrmX', '1-EzycNac2foljyynuQ6yp7XOLoRnGgOw', '1-FqO4CuNjbO1cgco10WWP2-wvpfumWJA', '1-GGrZnlwRvM-WbolJN5ekf1cNUxJVD9i', '1-ITRbltgdRrP69vgvE82G90iHEfRtPu-', '1-JELW6IiSuMnY5s7ZpCTyUvopz-xyOwv', '1-9AXkU-FSB4OlUfZ7AT-im0qxVuidc6N', '1-ASF9miMbVhiKOuaX_UDju7A22cYBoq1', '1-Ag6NU0WUBoqLy2LMuQdPWhRs6Ccf2oQ', '1-B4JrWCu6oADtX4bQEoS3iXoz51qXJAX', '1-BBpoKtBRjBC7ZZzD8S2FNr0VYWj1MVI', '18iJObs6_9G8qup0gjCpY7wHMtgchAQJ4', '1-N3t023ACyWQIOChNkkvHezM6lK6n8mz', '1-JtPGNKRPaiSB1P7KX2Y8tqiqU9k2L_M', '1--VZk-Q5QUw6nryG4Lj3XoCFbo0tyoCV', '1-1AUupj4lTkycxLSK3vToJuXxyQvVUQt', '1-1O68CLoaOAynEvFVXLRaBRe3yizV8q0', '1-2iXKnXHT930cXczh7Fx2r06GAzC01JG', '1-35rsQEUfRTkHolPDohQnEsg_oPICtl3', '1-3qrZqokDp7AEUj8X1ykzKF3019QDZN7', '1-55oNWNPv2uIUYwninEyhew135wpv7jY', '1-5Bl3NM8K1Qen6EC3U9ZCpiYmPlNflKE', '1-6v6qYAFKj8p0aZ1gW4gWjJVvzT6LuE0', '1-6xNuWy0waUtb7sxVnL8qdn5_1y52iRV', '1R-oEvM9zMaIeNr1SelEhv3vqWQd5tfum', '1G3FRYBECjo7lAHzh__Dgm0uy1U7df0Bo', '1v2kByrWqnnecY-fvluBc_WfbkU5kWznt', '1WfzfVMCsFkv3XXMNIny6adbGiP9gTAqf', '1-Okr8GbXX9qw9vXUffw7y5Q7XJTJsvKp', '1-L7s8Dse_mg0qL-022w7JYuSenkHIF6d', '1-7QghWZ6XlUaP3og-RsIO8hLaHvrS8fy']
S2_EXTRACT_PATCHES = ['1-UWHNUzyjdPZ_BAywG8SzJi-dWKR36fn', '1-TpLqdPGDowHmGX64UTqjDSPQlKxF9xl', '1-TMXIkr8rom6Wm5CAN2A3_cchQqG5Yp0', '1-Ud5AAy-EMEwV3LSLsVYguJdeBrIahMp', '1-VHVdudXeo5HqQwksm4ItxgoPA8BSZk8', '1-YxMXXfr76YoR3zXkugKON1spy69qMVF', '1-WwO-3HK-ad8NAQ9aGDCXvzfohNzyPgi', '1-WJqzCDCVnP2fr072Q8yRgxvnedj2a1f', '1-cCCrb5-TwS18MRHkdDYC3O9AwqjKd5s', '1-ZKTQ5aafdptlN3OHPFqUf-qowjHrEhW', '1-hgO_V4_obhroTe8zs7vJrM4OrFcYwjK', '1-d7k_2O0iZalaxtwyqdRniOnXgH9ay85', '1-ca0VRFpPfo1IUbD9gVptoUNTCZwYi0E', '1-hxZudOEHeCCB44jEvJ_-_bfk6wTJwp3', '1-j4rcf498GDEbMADSCAmuvPcRISuGSw3', '1300DzM999lg6MNPs_xOUwwVEWSSxducK', '1-83mvGq7rjuCo-JGq_MrgrKrWHxjDFhP', '1-9wgVm5JmwZrlj9kE9z9toTG4HgfZTQK', '1-FcrqgK49Cu0IxSWTeFKZHRw6uHaDOMR', '1-Gf6aTlXcvO4cF-GLWGUvNx_ZmnQM4op', '1-Lg2bAyo7nXuZXIDqabzVj-dsVPBv4Hf', '1-NNwrci-XO9RLid_xfe5Jx2VfBi3b5xc', '1-QNaV5dsKUb9660pOrWgKHq6Ofmu98dj', '1-S79QMRCuv1QnthOVpOys2r49_2JZHFM', '1-SV47DzTFO643geylf-im3FMDb_OmO1l', '1-Yu1FPP5y_bDM12HsBxqLSxc-uUFsWRs', '1-rbywltdc8xf4HDhPclqdjfZhP87JNjw', '1-xbotxCBzlzHb-H700MRAlWUY9lgoseD', '105Z1M7RbJ0Kv0voQIBUnGfy6zXh0UPnQ', '108RzXn9LSoFKjg1MhiTE822IpQ1mpwfJ', '10NklkqgFc_JaFHs9kXUY5ZnRXh-CrX1H', '10SE9ee6sPL7_6Cb7-rKSBH8L-R3PnlB8', '10TPBkRvjWo-Ltk90IPd8lBxa4v7Tk5xO', '10UDNkojo1PfJshn8dLeeBXUA_IBZIpYl', '10VGAlYyZrnzWRvCu9PeEl29IwJge_EUu', '10bagE1K0ZlRq_1uXr_vSyNEHvSLASqFE', '10mGzVviT3IYodILrXeIqlwLKB-SyBwVF', '10uZm-Nhlg-_z8cP5uTsJZVVxTQ8baXGy', '10wAaYhyufwsgEsTWFbJ6CXXZS8VCqpNY', '10yK65yUXNeaESWBPJuUoZ6AUF_Ua10mY', '110Pg7KVCV6VkJZlYswa-nHwlKPftoJFd', '110_tjy1V1F1z1z8iZn_2AQtijF7GKfYp', '119ME-r-qnqC2h7gIwNJP95yA1ZldMJxn']

CREATE_PATCH_FID = S2_CREATE_PATCHES[BATCH_NUMBER]
EXTRACT_PATCH_FID = S2_EXTRACT_PATCHES[BATCH_NUMBER]

!echo -e "[B{BATCH_NUMBER}-S2] Downloading Create Patches Files..."
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{CREATE_PATCH_FID}?alt=media -o output_{BATCH_NUMBER}_create_patches.zip

!echo -e "\n[B{BATCH_NUMBER}-S2] Unzipping Create Patches Files..."
!mkdir output_{BATCH_NUMBER}
!unzip -q output_{BATCH_NUMBER}_create_patches.zip -d output_{BATCH_NUMBER}

!echo -e "[B{BATCH_NUMBER}-S2] Downloading Extract Patches Files..."
!curl -H "Authorization: Bearer {ACCESS_TOKEN}" https://www.googleapis.com/drive/v3/files/{EXTRACT_PATCH_FID}?alt=media -o output_{BATCH_NUMBER}_extract_patches.zip

!echo -e "\n[B{BATCH_NUMBER}-S2] Unzipping Extract Patches Files..."
!unzip -q output_{BATCH_NUMBER}_extract_patches.zip -d output_{BATCH_NUMBER}

!echo -e "\n[B{BATCH_NUMBER}-S2] Deleting Extra Files..."
!rm -rf output_{BATCH_NUMBER}_create_patches.zip
!rm -rf output_{BATCH_NUMBER}_extract_patches.zip

!python "prostate-cancer-thesis/scripts/extract_features.py" --encoder "{ENCODER}" --output-base-directory "output_{BATCH_NUMBER}" --dataset-base-directory "prostate-cancer-thesis" --dataset-info-file-name "train.csv" --patch-size 512

!rm -rf output_{BATCH_NUMBER}/create_patches
!rm -rf output_{BATCH_NUMBER}/extract_patches

!echo -e "\n[B{BATCH_NUMBER}-S2] Creating ZIP File For Batch # {BATCH_NUMBER}"
!cd output_{BATCH_NUMBER}; zip -qr -1 "output_{BATCH_NUMBER}.zip" . ; mv "output_{BATCH_NUMBER}.zip" .. ; echo "ZIP File Created For Batch # {BATCH_NUMBER}"

!echo -e "\n[B{BATCH_NUMBER}-S2] Removing Directory For Batch # {BATCH_NUMBER}"
!rm -r "output_{BATCH_NUMBER}"; echo "Directory Removed For Batch # {BATCH_NUMBER}"

!echo -e "\n[B{BATCH_NUMBER}-S2] All Done!"