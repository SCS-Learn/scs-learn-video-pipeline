from playwright.sync_api import sync_playwright
import os
import json
import time
from dotenv import load_dotenv
import requests
import argparse

"""
Log into Panopto via CMU SSO/Duo in a browser and return a requests
Session pre-loaded with the resulting authenticated cookies.
"""

# Panopto's forms-auth cookie. Its arrival is what "logged in" means here --
# csrfToken is set before login too, so waiting on that would hand back a
# session that only fails later, in get_folder_id, with a confusing error.
AUTH_COOKIE = ".ASPXAUTH"
LOGIN_TIMEOUT = 600.0


def _auth_cookie(context):
    for c in context.cookies():
        if c["name"] == AUTH_COOKIE and "panopto.com" in c["domain"]:
            return c
    return None


def get_cookies(timeout=LOGIN_TIMEOUT):
    load_dotenv()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://scs.hosted.panopto.com")
        # CMU's SSO login form is brittle to script (field names/labels change),
        # so log in by hand in the browser window that just opened.
        sso_id = os.getenv("SSO_ID")
        print("\n" + "=" * 70)
        print("A browser window opened. Complete CMU login there:")
        print("  1. Click 'Sign in'")
        print(f"  2. Enter your Andrew ID ({sso_id or '<set SSO_ID in .env>'}) + password")
        print("  3. Approve the Duo push on your phone")
        print("Nothing to do here -- this waits for the login to land and then")
        print("carries on by itself.")
        print("=" * 70, flush=True)

        # This used to be a bare input(): press Enter when the page has loaded.
        # That made the stage need a TTY for no good reason -- run it anywhere
        # stdin is not a terminal (a `!` prefix in an agent session, a pipe, a
        # cron job) and it dies instantly with EOFError, having already opened
        # the browser. Polling for the cookie needs no stdin at all, and is
        # strictly more accurate besides: pressing Enter a second early handed
        # back a half-authenticated session, and pressing it late wasted time.
        deadline = time.time() + timeout
        cookie, last_note = None, 0.0
        while time.time() < deadline:
            cookie = _auth_cookie(context)
            if cookie:
                break
            waited = timeout - (deadline - time.time())
            if waited - last_note >= 15:
                last_note = waited
                print(f"[ingestion] waiting for SSO + Duo ... {waited:.0f}s "
                      f"of {timeout:.0f}s", flush=True)
            time.sleep(2)

        if not cookie:
            context.close()
            browser.close()
            raise RuntimeError(
                f"no {AUTH_COOKIE} cookie after {timeout:.0f}s -- the Panopto "
                f"login did not complete. Re-run and finish SSO + Duo in the "
                f"browser window, or raise --login-timeout.")

        print("[ingestion] authenticated", flush=True)
        page.wait_for_url("**://scs.hosted.panopto.com/**", timeout=60000)
        cookies = context.cookies()

        context.close()
        browser.close()

    session = requests.Session()
    for c in cookies:
        if "panopto.com" in c["domain"]:
            session.cookies.set(
                c["name"], c["value"], domain=c["domain"], path=c["path"]
            )

    return session


"""
Search Panopto folders for the course by semester and course number and return 
the matching folder's ID, raising if no match is found.
"""


def get_folder_id(search_term, session):
    csrf_token = session.cookies.get("csrfToken")
    if csrf_token is None:
        raise RuntimeError(
            "csrfToken missing — check that login/Duo actually completed."
        )

    response = session.get(
        "https://scs.hosted.panopto.com/Panopto/Api/Folders",
        params={
            "parentId": "null",
            "folderSet": 1,
            "searchTerm": search_term,
            "includeMyFolder": "false",
            "includePersonalFolders": "true",
            "page": 0,
            "sort": "Depth",
            "names[0]": "SessionCount",
        },
        headers={
            "x-csrf-token": csrf_token,
            "x-requested-with": "XMLHttpRequest",
            "accept": "application/json, text/javascript, */*; q=0.01",
        },
    )

    folders = response.json()
    if not folders:
        raise RuntimeError(f"No folder found matching {search_term}")
    folder_id = folders[0]["Id"]

    return folder_id


"""
Fetch the list of lecture sessions (up to 100) contained in the given
Panopto folder.
"""


def get_lectures(folder_id, session):
    csrf_token = session.cookies.get("csrfToken")
    if csrf_token is None:
        raise RuntimeError(
            "csrfToken missing — check that login/Duo actually completed."
        )

    response = session.post(
        "https://scs.hosted.panopto.com/Panopto/Services/Data.svc/GetSessions",
        json={"queryParameters": {"folderID": folder_id, "maxResults": 100}},
        headers={
            "x-csrf-token": csrf_token,
            "x-requested-with": "XMLHttpRequest",
            "content-type": "application/json; charset=utf-8",
        },
    )

    return response.json()


"""
Fetch asset info (video URLs, etc.) for a single lecture given its delivery ID.
"""


def get_lecture_asset(delivery_id, session):
    csrf_token = session.cookies.get("csrfToken")
    if csrf_token is None:
        raise RuntimeError(
            "csrfToken missing — check that login/Duo actually completed."
        )

    response = session.post(
        "https://scs.hosted.panopto.com/Panopto/Pages/Viewer/DeliveryInfo.aspx",
        data={"deliveryId": delivery_id, "responseType": "json"},
        headers={
            "x-csrf-token": csrf_token,
            "x-requested-with": "XMLHttpRequest",
        },
    )

    return response.json()


"""
Given the lectures list response, fetch and return the asset info for
every lecture in it, paired with the lecture's own session metadata.
"""


def get_assets(lectures, session):
    results = lectures["d"]["Results"]
    assets = []

    for result in results:
        d_id = result["DeliveryID"]
        asset = get_lecture_asset(d_id, session)
        assets.append((result, asset))

    return assets


"""
Build one manifest entry in the shape panopto_download.py expects:
"""


def build_lecture_entry(result, asset, course):
    delivery = asset.get("Delivery", asset)

    streams = []
    for s in delivery.get("Streams", []):
        url = s.get("StreamUrl")
        if not url:
            continue
        stream_type = "camera" if s.get("StreamType") == 1 else "screen"
        streams.append(
            {
                "type": stream_type,
                "isHls": s.get("ViewerMediaFileTypeName") == "hls",
                "url": url
            }
        )

    chapters = delivery.get("Timestamps", [])

    return {
        "key": f"{course}_{asset['SessionId']}",
        "id": asset["SessionId"],
        "name": delivery.get("SessionName"),
        "durationSec": delivery.get("Duration"),
        "course": course,
        "owner": delivery.get("OwnerDisplayName"),
        "start": delivery.get("SessionStartTime"),
        "chapters": chapters,
        "streams": streams,
    }


"""
Assemble the full manifest dict from the (result, asset) pairs and write
it to disk in the {"lectures": [...]} shape panopto_download.py reads.
"""


def build_manifest(assets, course, out_path="manifest.json"):
    lectures = [build_lecture_entry(result, asset, course) for result, asset in assets]
    manifest = {"lectures": lectures}

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semester", required=True, help='e.g. "Spring 2026"')
    parser.add_argument("--course", required=True, help='e.g. "10-301/601"')
    parser.add_argument("--out", default="manifest.json",
                        help="Where to write the manifest. Defaults to "
                             "manifest.json, which OVERWRITES any existing one "
                             "-- point this elsewhere when you already have a "
                             "manifest for another course.")
    parser.add_argument("--login-timeout", type=float, default=LOGIN_TIMEOUT,
                        help="Seconds to wait for SSO + Duo to complete")
    args = parser.parse_args()
    search_term = f"{args.semester}:  {args.course}"

    session = get_cookies(timeout=args.login_timeout)
    folder_id = get_folder_id(search_term, session)
    lectures = get_lectures(folder_id, session)
    assets = get_assets(lectures, session)
    build_manifest(assets, args.course, out_path=args.out)


if __name__ == "__main__":
    main()
