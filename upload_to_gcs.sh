#!/usr/bin/env bash
BUCKET=$1

download_unzip_and_upload() {
    # Zip files have different formats.
    # Sometimes fail to download data from website, if this is the case just try again later
    if wget https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_20$1.zip -O temp.zip ; then
        :
    else
        echo "ZIP file has different format"
        if wget https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q${2}_20$1.zip -O temp.zip ; then
            :
        else
            echo "[INFO] Data for 20$1 Q$2 not retrieved. Try again later"
        fi
    fi
    unzip temp.zip
    # Unnecessary directory sometimes made after unzipping donwloaded file
    if [ -d "__MACOSX/" ]; then
      # Take action if directory exists. #
      echo "Deleting unnecessary directory..."
      rm -rf __MACOSX
    fi
    FOLDER=$(unzip -qql temp.zip | head -n1 | tr -s ' ' | cut -d ' ' -f 5-)

    rm temp.zip
    # csv files are sometimes unzipped into a folder and sometimes just unzipped directly
    # into current directory. Either way, remove after adding to GCS
    if gsutil -m cp $FOLDER/*.csv gs://${BUCKET}/hard-drive-failure/${FOLDER} ; then {
        rm -rf $FOLDER
    }
    else {
        gsutil -m cp *.csv gs://${BUCKET}/hard-drive-failure/20$1
        rm *.csv
    }
    fi
}

for year in {13..15}
do
    download_unzip_and_upload $year
done

# Files change format after 2015
for year in {16..19}
do
    for quarter in {1..4}
    do
    if [[ $year -eq 19 ]] && [[ $quarter -eq 4 ]]; then
        continue
    fi
    download_unzip_and_upload $year $quarter
    done
done
