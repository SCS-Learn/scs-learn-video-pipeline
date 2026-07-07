from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv
import requests


def get_cookies():
    load_dotenv()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://scs.hosted.panopto.com")
        page.get_by_role("link", name="Sign in").click()

        sso_id = os.getenv("SSO_ID")
        sso_pass = os.getenv("SSO_PASSWORD")
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


def get_folder_id(session):
    response = session.get(
        "https://scs.hosted.panopto.com/Panopto/Api/Folders",
        params={
            "parentId": "null",
            "folderSet": 1,
            "searchTerm": "Spring 2026:  10-301/601 Introduction to Machine Learning",
            "includeMyFolder": "false",
            "includePersonalFolders": "true",
            "page": 0,
            "sort": "Depth",
            "names[0]": "SessionCount",
        },
        headers={
            "x-csrf-token": session.cookies.get("csrfToken"),
            "x-requested-with": "XMLHttpRequest",
            "accept": "application/json, text/javascript, */*; q=0.01",
        },
    )

    folders = response.json()
    if not folders:
        raise RuntimeError("No folder found matching search term.")
    folder_id = folders[0]["Id"]

    return folder_id


def get_lectures(folder_id, session):
    response = session.post(
        "https://scs.hosted.panopto.com/Panopto/Services/Data.svc/GetSessions",
        json={"queryParameters": {"folderID": folder_id, "maxResults": 100}},
        headers={
            "x-csrf-token": session.cookies.get("csrfToken"),
            "x-requested-with": "XMLHttpRequest",
            "content-type": "application/json; charset=utf-8",
        },
    )

    return response.json()


def main():
    cookies = get_cookies()
    folder_id = get_folder_id(cookies)
    lectures = get_lectures(folder_id, cookies)


if __name__ == "__main__":
    main()
