from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv

load_dotenv()

with sync_playwright() as p:
    browser = p.chromium.launch(headless = False)
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
    cookies = context.cookies()

    context.close()
    browser.close()

print([c["name"] for c in cookies])