from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv
import requests

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
        session.cookies.set(c["name"], c["value"], domain=c["domain"], path=c["path"])

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

print(response.status_code)
print(response.text[:1000])
