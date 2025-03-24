BATCH_NUMBER = 0

S2_CREATE_PATCHES = ['1-L91eWihyI4Tsow1rRtfendzD8pJGwfg', '1-MI26s6e00bQ1sfVgvVyU4KGrRHvucxo', '1-N4I8hhiZq_sYQi0xL3qM2vxcP7gGqWL', '1-OFN5S_e1rpb6sqo1JIjX6rsewq2IOm_', '1-PrMgOGfLiw8LZ0hISRd9fZwQZizIAHW', '1-QBffAnNpcbRw6AWebBpiInvtMBSPBVT', '1-QPTvMF0wMVSYI3SdfOVm5Wnz9f2VxHw', '1-R8HKZDJBT7eqhryAM14be7nqNkoURON', '1-RZLJjyjfw8rzlstTUfn2yz8hUz84DpF', '1-L8fCvS4QW2VLw1sXFYH7uBpv2gLF9AR', '1-CA0bhg2vHJHnTtZtEmk6t_BmVQ_C5xN', '1-DM8YROZ_uQE6cMvDIlQC2m-JQay-fkr', '1-E9j1vMFYjmIqYcWD-PguSpLtzjfgrmX', '1-EzycNac2foljyynuQ6yp7XOLoRnGgOw', '1-FqO4CuNjbO1cgco10WWP2-wvpfumWJA', '1-GGrZnlwRvM-WbolJN5ekf1cNUxJVD9i', '1-ITRbltgdRrP69vgvE82G90iHEfRtPu-', '1-JELW6IiSuMnY5s7ZpCTyUvopz-xyOwv', '1-9AXkU-FSB4OlUfZ7AT-im0qxVuidc6N', '1-ASF9miMbVhiKOuaX_UDju7A22cYBoq1', '1-Ag6NU0WUBoqLy2LMuQdPWhRs6Ccf2oQ', '1-B4JrWCu6oADtX4bQEoS3iXoz51qXJAX', '1-BBpoKtBRjBC7ZZzD8S2FNr0VYWj1MVI', '18iJObs6_9G8qup0gjCpY7wHMtgchAQJ4', '1-N3t023ACyWQIOChNkkvHezM6lK6n8mz', '1-JtPGNKRPaiSB1P7KX2Y8tqiqU9k2L_M', '1--VZk-Q5QUw6nryG4Lj3XoCFbo0tyoCV', '1-1AUupj4lTkycxLSK3vToJuXxyQvVUQt', '1-1O68CLoaOAynEvFVXLRaBRe3yizV8q0', '1-2iXKnXHT930cXczh7Fx2r06GAzC01JG', '1-35rsQEUfRTkHolPDohQnEsg_oPICtl3', '1-3qrZqokDp7AEUj8X1ykzKF3019QDZN7', '1-55oNWNPv2uIUYwninEyhew135wpv7jY', '1-5Bl3NM8K1Qen6EC3U9ZCpiYmPlNflKE', '1-6v6qYAFKj8p0aZ1gW4gWjJVvzT6LuE0', '1-6xNuWy0waUtb7sxVnL8qdn5_1y52iRV', '1R-oEvM9zMaIeNr1SelEhv3vqWQd5tfum', '1G3FRYBECjo7lAHzh__Dgm0uy1U7df0Bo', '1v2kByrWqnnecY-fvluBc_WfbkU5kWznt', '1WfzfVMCsFkv3XXMNIny6adbGiP9gTAqf', '1-Okr8GbXX9qw9vXUffw7y5Q7XJTJsvKp', '1-L7s8Dse_mg0qL-022w7JYuSenkHIF6d', '1-7QghWZ6XlUaP3og-RsIO8hLaHvrS8fy']

FID = S2_CREATE_PATCHES[BATCH_NUMBER]

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