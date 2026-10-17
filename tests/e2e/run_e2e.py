"""
Browser end-to-end checks (Playwright + local Chrome).

Starts the app from a temporary copy WITHOUT .streamlit/secrets.toml, so no
Gemini request can be made. Model files are symlinked in. Uses live
services: Google Drive (Cotton refresh), Open-Meteo (weather, geocoding)
and OpenStreetMap tiles. Device location is simulated by the browser.

    .venv/bin/python tests/e2e/run_e2e.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
SHOTS = ROOT / ".impeccable" / "review"
PORT = 8611
URL = f"http://localhost:{PORT}"
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(("PASS " if condition else "FAIL ") + name, detail, flush=True)


def start_app():
    workdir = Path(tempfile.mkdtemp(prefix="agri-e2e-"))
    for name in ("app.py", "agri_core.py"):
        shutil.copy(ROOT / name, workdir / name)
    (workdir / ".streamlit").mkdir()
    shutil.copy(ROOT / ".streamlit" / "config.toml", workdir / ".streamlit" / "config.toml")
    for model in ("crop_router_v2", "corn_model", "cotton_doctor_latest"):
        os.symlink(ROOT / f"{model}.tflite", workdir / f"{model}.tflite")
    assert not (workdir / ".streamlit" / "secrets.toml").exists()
    process = subprocess.Popen(
        [str(ROOT / ".venv" / "bin" / "streamlit"), "run", "app.py",
         "--server.headless", "true", "--server.port", str(PORT)],
        cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(60):
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2).read() == b"ok":
                return process, workdir
        except Exception:
            time.sleep(1)
    raise RuntimeError("app did not start")


def open_page(browser, width=1440, height=900, **context):
    ctx = browser.new_context(viewport={"width": width, "height": height}, **context)
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("text=Read a leaf", timeout=60000)
    return ctx, page


def upload(page, filename):
    page.set_input_files("input[type=file][accept*='.jpg']", str(DATA / filename))
    page.wait_for_selector("text=Run diagnosis", timeout=120000)
    page.wait_for_timeout(1500)


def main_text(page):
    return page.inner_text("[data-testid=stMain]")


def sidebar(page):
    return page.inner_text("[data-testid=stSidebar]")


def leaflet(page):
    for frame in page.frames:
        try:
            if frame.locator(".leaflet-container").count():
                return frame
        except Exception:
            pass


def diagnosis_checks(browser):
    ctx, page = open_page(browser)
    upload(page, "rice_brown_spot.jpg")
    text = main_text(page)
    match = re.search(r"ROUTER SUGGESTS[^\n]*", text)
    check("rice photo: router suggestion shown with status", match and "NEEDS REVIEW" in match.group(0), match.group(0) if match else "")
    check("rice photo: confirm-crop prompt", "Confirm the crop" in text)
    page.screenshot(path=str(SHOTS / "e2e-rice-review.png"))
    page.get_by_text("Rice", exact=True).click()
    page.wait_for_timeout(1500)
    check("rice chosen: SET MANUALLY", "SET MANUALLY" in main_text(page))
    page.get_by_role("button", name="Run diagnosis").click()
    page.wait_for_selector(".verdict", timeout=240000)
    page.wait_for_timeout(1500)
    text = main_text(page)
    verdict = re.search(r"(Disease:|Weed:|Leaf reads|Uncertain)[^\n]*", text)
    check("rice: diagnosis runs with model from Drive (live)", verdict is not None, verdict.group(0) if verdict else "")
    check("rice: result marked not validated", "has not been validated" in text and "Needs review" in text)
    page.set_viewport_size({"width": 1440, "height": 3400})
    page.wait_for_timeout(800)
    page.screenshot(path=str(SHOTS / "e2e-rice-diagnosis.png"))
    ctx.close()

    ctx, page = open_page(browser)
    upload(page, "maize_common_rust.jpg")
    page.get_by_role("button", name="Run diagnosis").click()
    page.wait_for_selector(".verdict", timeout=180000)
    page.wait_for_timeout(1500)
    text = main_text(page)
    check("maize rust: verdict Common Rust", "Disease: Common Rust" in text)
    check("maize rust: 4 of 4 quadrants agree", "4 OF 4 QUADRANTS AGREE" in text)
    check("no key: prescription NOT WRITTEN, no key text", "NOT WRITTEN" in text and "no Gemini key is configured" in text and "sk-" not in text)
    check("no key: sidebar says prescriptions paused", "Prescriptions paused" in sidebar(page))
    ctx.close()

    ctx, page = open_page(browser)
    upload(page, "maize_nclb.jpg")
    page.get_by_role("button", name="Run diagnosis").click()
    page.wait_for_selector(".verdict", timeout=180000)
    page.wait_for_timeout(1500)
    text = main_text(page)
    check("maize NLB: whole-leaf verdict Blight", "Disease: Blight" in text)
    check("maize NLB: needs review + quadrant-only note", "Needs review" in text and "Seen only in individual quadrants" in text)
    page.set_viewport_size({"width": 1440, "height": 3400})
    page.wait_for_timeout(800)
    page.screenshot(path=str(SHOTS / "e2e-maize-nclb.png"))
    ctx.close()

    ctx, page = open_page(browser)
    upload(page, "cotton_leaves_botgard.jpg")
    text = main_text(page)
    check("cotton: routed and ACCEPTED without review", "ACCEPTED" in text and "Confirm the crop" not in text)
    page.get_by_role("button", name="Run diagnosis").click()
    page.wait_for_selector(".verdict", timeout=240000)
    page.wait_for_timeout(1500)
    text = main_text(page)
    verdict = re.search(r"(Disease:|Weed:|Leaf reads|Uncertain)[^\n]*", text)
    check("cotton: diagnosis completed (live Drive refresh)", "QUADRANTS AGREE" in text, verdict.group(0) if verdict else "")
    ctx.close()

    ctx, page = open_page(browser)
    page.set_input_files("input[type=file][accept*='.jpg']", files=[{"name": "broken.jpg", "mimeType": "image/jpeg", "buffer": b"not really a jpeg"}])
    page.wait_for_timeout(2500)
    check("corrupt upload: readable error, no crash", "could not be read as an image" in main_text(page))
    ctx.close()

    ctx, page = open_page(browser, 390, 844)
    upload(page, "rice_brown_spot.jpg")
    page.set_viewport_size({"width": 390, "height": 2600})
    page.wait_for_timeout(800)
    check("mobile: no horizontal overflow", page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
    page.screenshot(path=str(SHOTS / "e2e-mobile-rice.png"))
    ctx.close()


def field_checks(browser):
    ctx, page = open_page(browser, 1440, 1000, geolocation={"latitude": 17.385, "longitude": 78.4867, "accuracy": 35}, permissions=["geolocation"])
    check("location: nothing shown before the request", "17.38500" not in sidebar(page))
    page.get_by_role("button", name="Detect my location").click()
    page.wait_for_selector("text=17.38500", timeout=30000)
    page.wait_for_timeout(2000)
    text = sidebar(page)
    check("location granted: coordinates, source, accuracy", "17.38500, 78.48670" in text and "device's location" in text and "±35 m" in text)
    reading = re.search(r"Forecast reading[^\n]*", text)
    check("live weather: soil moisture reading or explicit unavailable", reading or "Weather service unavailable" in text, reading.group(0) if reading else "")
    page.get_by_text("Field location and map").click()
    page.wait_for_timeout(3500)
    frame = leaflet(page)
    check("map rendered with field marker (live OSM tiles)", frame is not None and frame.locator(".leaflet-marker-icon").count() >= 1)
    box = frame.locator(".leaflet-container").bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.3)
    page.wait_for_timeout(3500)
    check("map click corrects the field point", "set on the map" in sidebar(page))
    page.get_by_text("Clicking the map moves the field point").click()
    page.wait_for_timeout(1500)
    frame = leaflet(page)
    frame.locator(".leaflet-draw-draw-polygon").click()
    box = frame.locator(".leaflet-container").bounding_box()
    for x, y in [(0.45, 0.45), (0.62, 0.45), (0.62, 0.62), (0.45, 0.62), (0.45, 0.45)]:
        page.mouse.click(box["x"] + box["width"] * x, box["y"] + box["height"] * y)
        page.wait_for_timeout(350)
    page.wait_for_timeout(4000)
    text = main_text(page)
    area = re.search(r"Outlined area[^\n]*", text)
    check("polygon boundary: area shown, labelled as drawn by you", area and "drawn by you" in area.group(0), area.group(0) if area else "")
    page.get_by_label("Field name").fill("North paddy")
    page.get_by_role("button", name="Save field").click()
    page.wait_for_timeout(2000)
    text = main_text(page)
    check("field saved with boundary; JSON download offered", "North paddy" in text and "boundary saved" in text and page.get_by_role("button", name="Download saved fields (JSON)").count() == 1)
    ctx.close()

    ctx, page = open_page(browser)
    page.get_by_role("button", name="Detect my location").click()
    try:
        page.wait_for_selector("text=Location permission was denied", timeout=30000)
        denied = True
    except Exception:
        denied = False
    check("location denied: clear message, no coordinates", denied and "device's location" not in sidebar(page))
    ctx.close()

    ctx, page = open_page(browser)
    box = page.get_by_label("Farm location")
    box.fill("Warangal")
    box.press("Enter")
    page.wait_for_timeout(4000)
    text = sidebar(page)
    check("place search (live geocoder)", "Warangal" in text and "from place search" in text)
    box.fill("zzqqxxnotaplace")
    box.press("Enter")
    page.wait_for_timeout(4000)
    check("place search: no-match message", "No place found" in sidebar(page))
    ctx.close()


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    process, workdir = start_app()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
            diagnosis_checks(browser)
            field_checks(browser)
            browser.close()
    finally:
        process.terminate()
        shutil.rmtree(workdir, ignore_errors=True)
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} browser checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
