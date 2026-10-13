"""Check the dashboard; install the research snapshot's [dashboard] extra first."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ["FUZEV_ROOT"]).expanduser().resolve()
OUT = ROOT / "runs/dashboard"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    base_url = f"http://127.0.0.1:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for name, width, height in [("desktop", 1440, 1000), ("mobile", 390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base_url)
            page.wait_for_function("document.getElementById('connection').textContent === 'Live'")
            state = page.request.get(base_url + "/api/state").json()
            active_parents = [task for task in state["tasks"] if task["active_descendants"]]
            for task in active_parents:
                summary = page.locator("summary").filter(
                    has=page.get_by_text(task["title"], exact=True)
                )
                assert summary.locator(".badge").text_content() == "In progress"
            page.locator("summary").first.click()
            page.wait_for_function("document.querySelector('details').open")
            page.wait_for_timeout(1200)
            assert page.locator("details").first.evaluate("el => el.open")
            layout = page.evaluate("""() => ({
                width: document.documentElement.scrollWidth, viewport: innerWidth,
                background: getComputedStyle(document.documentElement).backgroundColor,
                plotted: [...document.getElementById('chart').getContext('2d').getImageData(0,0,32,320).data].some((x,i)=>i%4===3&&x>0)
            })""")
            assert layout["width"] <= width, layout
            assert layout["background"] == "rgb(22, 25, 28)", layout
            assert layout["plotted"], layout
            assert not errors, errors
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            results.append({"viewport": name, "passed": True, "layout": layout,
                            "active_parents": [task["id"] for task in active_parents],
                            "errors": errors})
            page.close()
        browser.close()
    (OUT / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
