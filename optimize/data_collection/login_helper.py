"""Open a visible Chrome window to log in to a recruitment site.

Saves session cookies to a Chrome profile directory so subsequent crawler
runs can reuse them without going through login again.

Usage
-----
    python -m optimize.data_collection.login_helper --site lagou
    python -m optimize.data_collection.login_helper --site lagou --profile optimize/.chrome_profile
"""

from __future__ import annotations

import argparse
import pathlib

_SITE_URLS = {
    "lagou": "https://www.lagou.com",
    "boss":  "https://www.zhipin.com",
}


def open_login_window(site: str, profile_dir: str) -> None:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    profile = str(pathlib.Path(profile_dir).resolve())
    url = _SITE_URLS.get(site, site)

    opts = Options()
    opts.add_argument(f"--user-data-dir={profile}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )

    print(f"Opening {url} ...")
    driver.get(url)
    print()
    print("Steps:")
    print("  1. Complete the slider CAPTCHA in the browser window.")
    print("  2. Log in with your account.")
    print("  3. Return here and press Enter.")
    print()
    input("Press Enter after login is complete ...")
    driver.quit()
    print(f"Session saved to: {profile}")
    print("You can now run the crawler with --chrome-profile", profile)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", default="lagou", choices=list(_SITE_URLS),
                   help="Target site (default: lagou)")
    p.add_argument("--profile", default="optimize/.chrome_profile",
                   help="Chrome user-data-dir to save cookies (default: optimize/.chrome_profile)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    open_login_window(site=args.site, profile_dir=args.profile)
