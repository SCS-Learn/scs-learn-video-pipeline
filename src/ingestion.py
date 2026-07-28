from playwright.sync_api import sync_playwright
import os
import json
from dotenv import load_dotenv
import requests
import argparse
import getpass

"""
Log into Panopto via CMU SSO/Duo in a browser and return a requests
Session pre-loaded with the resulting authenticated cookies.
"""


def get_cookies():
    load_dotenv()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://scs.hosted.panopto.com")
        page.get_by_role("link", name="Sign in").click()

        sso_id = os.getenv("SSO_ID")
        sso_pass = getpass.getpass("Enter your SSO password: ")
        page.get_by_role("textbox", name="AndrewID").fill(f"{sso_id}")
        page.locator("#passwordinput").fill(f"{sso_pass}")
        page.get_by_role("button", name="Login").click()
        print("Approve the Duo push now, then press Enter here.")
        input()

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
    args = parser.parse_args()
    search_term = f"{args.semester}:  {args.course}"

    session = get_cookies()
    folder_id = get_folder_id(search_term, session)
    lectures = get_lectures(folder_id, session)
    assets = get_assets(lectures, session)
    build_manifest(assets, args.course, out_path="manifest.json")


if __name__ == "__main__":
    main()
