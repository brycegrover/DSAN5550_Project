#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Clean AU_SI12 Downloader
#   - Downloads ONLY .he5 data files
#   - Filters by year and month
#   - Stores outputs in a clear directory
#   - Leaves NSIDC authentication + CMR logic unchanged
# ---------------------------------------------------------------------------

from __future__ import print_function
import base64, getopt, itertools, json, math, netrc, os, ssl, sys, time
from getpass import getpass

try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import (
        urlopen, Request, HTTPError, URLError, build_opener, HTTPCookieProcessor
    )

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

short_name = "AU_SI12"
version = "1"

# Time range – adjust freely
time_start = "2023-01-01T00:00:00Z"
time_end   = "2025-11-12T23:59:59Z"

# Spatial region – Arctic only
bounding_box = "-180,65,180,90"
polygon = ""
filename_filter = ""
url_list = []

# Directory to save downloaded files
download_dir = "/Users/brycegrover/Desktop/DSAN/FALL_2025/DSAN5550/Project/data"

# Download only these year-month patterns
# Adjust to your needs
keep_patterns = [
    "202306","202307","202308","202309","202310",
    "202406","202407","202408","202409","202410",
    "202506","202507","202508","202509","202510"
]

# ---------------------------------------------------------------------------
CMR_URL = "https://cmr.earthdata.nasa.gov"
URS_URL = "https://urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = (
    "{0}/search/granules.json?"
    "&sort_key[]=start_date&sort_key[]=producer_granule_id"
    "&page_size={1}".format(CMR_URL, CMR_PAGE_SIZE)
)
CMR_COLLECTIONS_URL = "{0}/search/collections.json?".format(CMR_URL)
FILE_DOWNLOAD_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# AUTH + UTILS
# ---------------------------------------------------------------------------

def get_username():
    try: do_input = raw_input
    except NameError: do_input = input
    return do_input("Earthdata username (or press Return for token): ")

def get_password():
    password = ""
    while not password:
        password = getpass("password: ")
    return password

def get_token():
    token = ""
    while not token:
        token = getpass("bearer token: ")
    return token

def get_login_credentials():
    credentials = None
    token = None

    try:
        info = netrc.netrc()
        username, _acc, password = info.authenticators(urlparse(URS_URL).hostname)
        if username == "token":
            token = password
        else:
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    except:
        username = None

    if not username:
        username = get_username()
        if username:
            password = get_password()
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        else:
            token = get_token()

    return credentials, token

def build_version_query_params(version):
    version = str(int(version))
    out = ""
    desired = 3
    while len(version) <= desired:
        out += "&version=" + version.zfill(desired)
        desired -= 1
    return out

def build_filename_filter(filename_filter):
    if not filename_filter:
        return ""
    filters = filename_filter.split(",")
    out = "&options[producer_granule_id][pattern]=true"
    for f in filters:
        f = f if f.startswith("*") else "*" + f
        f = f if f.endswith("*") else f + "*"
        out += "&producer_granule_id[]=" + f
    return out

def build_query_params_str(
    short_name, version, time_start="", time_end="", bounding_box=None,
    polygon=None, filename_filter=None, provider=None
):
    params = "&short_name=" + short_name
    params += build_version_query_params(version)
    if time_start or time_end:
        params += f"&temporal[]={time_start},{time_end}"
    if polygon:
        params += "&polygon=" + polygon
    elif bounding_box:
        params += "&bounding_box=" + bounding_box
    if filename_filter:
        params += build_filename_filter(filename_filter)
    if provider:
        params += "&provider=" + provider
    return params

def build_cmr_query_url(
    short_name, version, time_start, time_end,
    bounding_box=None, polygon=None, filename_filter=None, provider=None
):
    params = build_query_params_str(
        short_name, version, time_start, time_end,
        bounding_box, polygon, filename_filter, provider
    )
    return CMR_FILE_URL + params

def get_speed(elapsed, chunk):
    if elapsed <= 0: return ""
    speed = chunk / elapsed
    units = ("","k","M","G","T","P")
    i = int(math.floor(math.log(speed, 1000))) if speed>0 else 0
    return f"{speed/(1000**i):.1f}{units[i]}B/s"

def output_progress(count,total,status="",bar_len=60):
    if total <= 0: return
    frac = min(max(count/float(total), 0), 1)
    filled = int(round(bar_len*frac))
    pct = int(round(100*frac))
    bar = "="*filled + " "*(bar_len-filled)
    s = f"  [{bar}] {pct:3d}%  {status}   "
    print("\b"*(len(s)+4), end="")
    sys.stdout.write(s)
    sys.stdout.flush()

def cmr_read_in_chunks(file_obj, chunk_size=1024*1024):
    while True:
        data = file_obj.read(chunk_size)
        if not data:
            break
        yield data

def get_login_response(url, credentials, token):
    opener = build_opener(HTTPCookieProcessor())
    req = Request(url)

    if token:
        req.add_header("Authorization", f"Bearer {token}")
    elif credentials:
        try:
            response = opener.open(req)
            url = response.url
        except HTTPError:
            pass
        req = Request(url)
        req.add_header("Authorization", f"Basic {credentials}")

    try:
        return opener.open(req)
    except HTTPError as e:
        if "Unauthorized" in e.reason:
            print("Unauthorized. Check username/password or token.")
            sys.exit(1)
        raise

# ---------------------------------------------------------------------------
# CMR SEARCH + DOWNLOAD
# ---------------------------------------------------------------------------

def check_provider_for_collection(short_name, version, provider):
    q = build_query_params_str(short_name,version,provider=provider)
    url = CMR_COLLECTIONS_URL + q
    try:
        resp = urlopen(Request(url))
    except Exception as e:
        print("Provider check error:", e); sys.exit(1)

    data = json.loads(resp.read().decode())
    return ("feed" in data and "entry" in data["feed"]
            and len(data["feed"]["entry"])>0)

def get_provider_for_collection(short_name, version):
    for provider in ["NSIDC_CPRD","NSIDC_ECS"]:
        if check_provider_for_collection(short_name, version, provider):
            return provider
    raise RuntimeError("No provider found for collection")

def cmr_filter_urls(search_results):
    if "feed" not in search_results or "entry" not in search_results["feed"]:
        return []

    entries = [e["links"] for e in search_results["feed"]["entry"] if "links" in e]
    links = list(itertools.chain(*entries))

    urls = []
    seen = set()
    for link in links:
        href = link.get("href")
        if not href: continue
        if link.get("inherited") is True: continue
        if "rel" in link and "data#" not in link["rel"]: continue
        fn = href.split("/")[-1]
        if "metadata#" in link.get("rel","") and fn.endswith(".dmrpp"): continue
        if "metadata#" in link.get("rel","") and fn == "s3credentials": continue
        if fn in seen: continue
        seen.add(fn)
        urls.append(href)

    return urls

def cmr_search(short_name, version, time_start, time_end,
               bounding_box="", polygon="", filename_filter="", quiet=False):

    provider = get_provider_for_collection(short_name, version)
    cmr_url = build_cmr_query_url(
        short_name, version, time_start, time_end,
        bounding_box, polygon, filename_filter, provider
    )

    if not quiet:
        print("Querying:\n ", cmr_url, "\n")

    paging_header = "cmr-search-after"
    page_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    urls = []
    hits = 0

    while True:
        req = Request(cmr_url)
        if page_id:
            req.add_header(paging_header, page_id)

        resp = urlopen(req, context=ctx)
        headers = {k.lower():v for k,v in dict(resp.info()).items()}

        if not page_id:
            hits = int(headers["cmr-hits"])
            print("Found", hits, "matches.")

        page_id = headers.get(paging_header)

        page = json.loads(resp.read().decode())
        new_urls = cmr_filter_urls(page)
        if not new_urls:
            break
        urls.extend(new_urls)

    return urls

def cmr_download(urls, force=False, quiet=False):
    if not urls:
        print("No files to download.")
        return

    os.makedirs(download_dir, exist_ok=True)

    print("Downloading", len(urls), "files...")

    credentials = None
    token = None

    for idx, url in enumerate(urls, start=1):
        if not credentials and not token:
            if url.startswith("https"):
                credentials, token = get_login_credentials()

        fname = url.split("/")[-1]
        path = os.path.join(download_dir, fname)

        print(f"{str(idx).zfill(len(str(len(urls))))}/{len(urls)}: {path}")

        for attempt in range(1, FILE_DOWNLOAD_MAX_RETRIES+1):
            try:
                resp = get_login_response(url, credentials, token)
                length = int(resp.headers["content-length"])

                if not force and os.path.exists(path):
                    if os.path.getsize(path) == length:
                        print("  File exists, skipping.")
                        break

                chunk_size = min(max(length,1), 1024*1024)
                max_chunks = int(math.ceil(length/chunk_size))
                count = 0
                t0 = time.time()

                with open(path,"wb") as f:
                    for data in cmr_read_in_chunks(resp,chunk_size):
                        f.write(data)
                        count += 1
                        if not quiet:
                            speed = get_speed(time.time()-t0, count*chunk_size)
                            output_progress(count, max_chunks, speed)
                print()
                break

            except Exception as e:
                print("Error downloading:", e)
                if attempt == FILE_DOWNLOAD_MAX_RETRIES:
                    print("Failed permanently:", fname)
                    sys.exit(1)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv=None):
    global url_list
    force=False; quiet=False

    if argv is None: argv=sys.argv[1:]

    try:
        opts,_ = getopt.getopt(argv,"hfq",["help","force","quiet"])
        for o,_ in opts:
            if o in ("-f","--force"): force=True
            elif o in ("-q","--quiet"): quiet=True
            elif o in ("-h","--help"):
                print("usage: python data_download.py [-f] [-q]")
                sys.exit(0)
    except:
        print("Bad args."); sys.exit(1)

    # Query
    url_list = cmr_search(
        short_name,version,
        time_start,time_end,
        bounding_box,polygon,filename_filter,
        quiet=quiet
    )

    # Filter to melt season & only .he5
    filtered = []
    for u in url_list:
        if u.endswith(".he5") and any(pat in u for pat in keep_patterns):
            filtered.append(u)

    print(f"After month + .he5 filtering: kept {len(filtered)} of {len(url_list)} URLs.")

    cmr_download(filtered, force=force, quiet=quiet)


if __name__ == "__main__":
    main()
