from typing import List
from glob import glob
from importlib import resources
import requests #type: ignore[import]
from netrc import netrc
from tqdm import tqdm #type: ignore[import]
import os


LOGIN_ENDPOINT = "urs.earthdata.nasa.gov"
NETRC_PATH = glob(str(resources.files(__package__) / "assets/.netrc"))[0]
OUTPUT_DIR = resources.files(__package__) / "qualitative/"

class EarthdataSession(requests.Session):
    def rebuild_auth(self, prepared_request, response):
        # if redirecting to urs.earthdata.nasa.gov, keep credentials
        if LOGIN_ENDPOINT in prepared_request.url:
            prepared_request.headers["Authorization"] = response.request.headers.get("Authorization")

def download_scenes(uris:List[str]) -> List[str]:
    """
    References: https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget
    """
    if len(uris) == 0: return []
    completed_transaction = 0 # this is much simpler than registering OS signal interupt callbacks

    # 1. set up the query
    auth_data = netrc(NETRC_PATH).authenticators(LOGIN_ENDPOINT)
    if auth_data is None:
        raise ValueError(f"No entry for {LOGIN_ENDPOINT} found in .netrc")
    username, _, password = auth_data

    session = EarthdataSession()
    session.auth = (username, password)
    session.trust_env = False

    # 2. query per uri
    outfiles = []
    for uri in tqdm(uris):
        of = str(uri).split('/')[-1]
        assert of.endswith('.nc'), f"Unable to parse the .nc output file for {uri}"
        out_fp = str(OUTPUT_DIR / of)

        if os.path.exists(out_fp):
            outfiles.append(out_fp)
            continue
        print(f"{of} not found. Downloading (may take a few minutes)...")
        try:
            with session.get(uri, stream=True, allow_redirects=True) as r:
                r.raise_for_status()

                with open(out_fp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"Completed {of}!")
            completed_transaction = 1
            outfiles.append(out_fp)
        except Exception as e:
            raise e
        finally:
            # if interupted, remove the file if it exists (could be a partial file)
            if completed_transaction == 0 and os.path.exists(out_fp):
                os.remove(out_fp)
            completed_transaction = 0
    return outfiles
