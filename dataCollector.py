import os
import os.path
import requests
import time

# only retry two times before going to the next item
requests.adapters.DEFAULT_RETRIES = 2


def download_data():
    """ Download (imagenet) data """

    dir_hotdogs = os.path.join('train', 'hotdogs')

    # get download urls from cow.txt
    download_urls = open('downloadList.txt',encoding='utf8').readlines()

    # hold number of files downloaded
    download_counter = 328

    # iterate over urls
    for download_url in download_urls:
        print("Downloading and storing file %s" % download_url.strip())

        try:
            # download the file
            req = requests.get(download_url.strip(), stream=True)
            target_path = os.path.join(dir_hotdogs, '%s.jpg' \
                                       % download_counter)

            # store file locally
            with open(target_path, 'wb') as image:
                for chunk in req.iter_content(1024):
                    image.write(chunk)

            download_counter += 1

        except requests.exceptions.RequestException as exception:
            print("Skipping file %s" % download_url.strip())

        # wait 5 second so we don't hammer servers
        time.sleep(5)


if __name__ == "__main__":
    download_data()
