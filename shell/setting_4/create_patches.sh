BATCH_NUMBER = 0

!python "prostate-cancer-thesis/scripts/create_patches.py" --patch-size 256 --stride-size 128 --save-masks True --stitching True --selective-slides True --selective-slides-csv "prostate-cancer-thesis/data_splits/train_{BATCH_NUMBER}.csv" --output-base-directory "output_{BATCH_NUMBER}"

!echo "Creating ZIP File For Batch # {BATCH_NUMBER}"
!cd output_{BATCH_NUMBER}; zip -qr -1 "output_{BATCH_NUMBER}.zip" . ; mv "output_{BATCH_NUMBER}.zip" .. ; echo "ZIP File Created For Batch # {BATCH_NUMBER}"

!echo "Removing Directory For Batch # {BATCH_NUMBER}"
!rm -r "output_{BATCH_NUMBER}"; echo "Directory Removed For Batch # {BATCH_NUMBER}"