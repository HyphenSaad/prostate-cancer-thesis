BATCH_NUMBER = 0

S3_CREATE_PATCHES = ['12wPf9U0kLygUdPsqhneVJBpJ0Uy0SMi2', '13iZVOBUu7Y-v-hqy4wTLnmCBQ-Evwvbw', '14Jx6filtfYYTZsIlijN_a8rcVteGkhRZ', '15-baV9dQ9Dia2gMVSwm3u2imuo2tSVtp', '158Xh48HJR2fr2s2XgCXf2UgmEQPWhBLR', '15DGhFUl7TEdAKj8BmVaWc4Ox4uUNekvz', '15J9bMGOXdsHUBdh4DlFmA_9r7YcgieJT', '15MwO_c4HB1-yJ8WgUG7QhyzOFv1HUOqG', '15Psrrf0dmHIyRcFJiaP4h9V7o3zVM0kc', '15ciocWFrJvZiWsqQ7ZGkBb3nGIxDsw8F', '12wb9urvUQNypsgIs_YW8MtHv_NUuXZ3o', '131mAhHp91hUYkYDpnZ4JHKzEgpONSW_J', '133fZL1NS2QJzKR82UCd49LRkEZPXZu9S', '13Boxp9aeTadWqzm9oWGbRc5kc8o63JNO', '13C2fyoExtwNXbQORLpyH2WsDKOcT0vIe', '13CN3XQDwuahEs6j8RFL8fBG5om0O-y44', '13HPMQG7I--zvHdF2JT2yhc7Z0GsmL1sg', '13OD1T2L_oiiCOiR2I9RNXTX4i38nmFXt', '13VRT5fs7yZqTP4C__0XVK4QMI-mfG6hg', '13cyqLFzD0bwpqU_2mHDOU0-QTGa6TGGU', '13iq0Il1Tb-omDaLiByrRMMyWSvDn3HM9', '13n-EYSwjStHjwAU2pOPaDBJRcfXTm0i8', '13nRRxPzxSxEceIRaLEZv1W7yaFYBUzik', '13t7n1aBkQxO8B1lblA94z31H_1ofttgW', '13vw8V_gYjmVLMYzvrjrDLFEuVDBif65V', '1434xVqT3FB1HW3fXMQVP8aLTiGkWM7QH', '144C-UpJuDD8RVSu86nlsA5rlO6JkWAAW', '14EMinyuaJSpvsD5APrVfxJ_yBrg3Jg0I', '14JSm1GK4NUoDGRNVCGRnXM0LgjZdnP29', '14JlvFIUPYj4aJv8aDBU_tVSrR1iI9QDH', '14LWWfNH-QSsuBF-1jQaB44-2jZXoElAa', '14Nt79kWArKw_aUH7i2KUjk5Y5wVzjUeJ', '14RUverayNZ96CUPNyzw_SzrFTlBpAQjO', '14g-aPP4bU9Aj99Inmdl6y6UthBzlhZNK', '14h6EksZ-LReToMQGou3vO3C2FjDDfoG-', '14hce4vacHRlMQjZNXYSjZtcL-VM3cjlk', '14ilPsvN8HEWkJPy5ks6TU6DJgKn-0QtZ', '14iniyhMGEUEyiJp8nVKFfUpf29gscygE', '14okVb6ILirlj6dUP6Kj6iMDPlgSSyrRa', '14yKm0Q7jvVKSc7DxWh4iWn6nWPYhbK7r', '151CrslMc7RsmWoqFcuk38sC9HUy5BEmR', '151Hn-URMdCdm5h4RBdkfySsyRevUOr-Y', '1599c7w4PSdu3nqRw52hv4UnyekTJ-6w8']

FID = S3_CREATE_PATCHES[BATCH_NUMBER]

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