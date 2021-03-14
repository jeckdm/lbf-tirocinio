import zipfile
import os


def extract_url(outer_zip_name, result_file_name, verbose=False):
    count = 0

    with open(result_file_name, 'wb') as out_file:

            zf = zipfile.ZipFile(outer_zip_name)
            for nested_zip in zf.namelist():
                if nested_zip[-4:] == '.zip':
                    if verbose:
                        print(f'processing {nested_zip}')
                    zf.extract(nested_zip)

                    nzf = zipfile.ZipFile(nested_zip)
                    for name in nzf.namelist():
                        if name[-7:] == 'URL.txt':
                            #print(name)
                            out_file.write(nzf.read(name))
                            out_file.write(b'\n')
                            count += 1
                    os.remove(nested_zip)


    if verbose:
        print(f'wrote {count} strings in {result_file_name}')
        
extract_url('Legitimate.zip', 'Legitimate.txt', verbose=True)
extract_url('Phishing.zip', 'Phishing.txt', verbose=True)
