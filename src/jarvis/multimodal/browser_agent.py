"""Browser session for JARVIS computer-use tasks.

This is a real browser automation seam (Playwright), not a mocked vision response.
JARVIS controls a dedicated persistent Firefox profile so it can inspect the live DOM,
find images/links, download files, and verify navigation. It does not silently attach
to an unrelated pre-existing Firefox window.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote_plus
from typing import Any


class BrowserAgent:
    def __init__(self, download_dir: Path | None = None) -> None:
        self.download_dir = download_dir or (Path.home() / "Downloads" / "JARVIS")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._playwright: Any = None
        self._context: Any = None
        self._page: Any = None

    def _ensure_browser(self) -> Any:
        if self._page is not None and not self._page.is_closed():
            return self._page
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Browser automation requires Playwright. Run 'uv sync --extra browser' "
                "and 'uv run playwright install firefox'."
            ) from exc
        self._playwright = sync_playwright().start()
        profile = Path.home() / ".jarvis" / "firefox-profile"
        profile.mkdir(parents=True, exist_ok=True)
        self._context = self._playwright.firefox.launch_persistent_context(
            user_data_dir=str(profile),
            headless=False,
            accept_downloads=True,
            downloads_path=str(self.download_dir),
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        return self._page

    @property
    def current_url(self) -> str:
        if self._page is None or self._page.is_closed():
            return ""
        return self._page.url

    def search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "What would you like me to search for?"
        page = self._ensure_browser()
        url = "https://www.google.com/search?q=" + quote_plus(query)
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(1200)
        title = page.title()
        if "/sorry/" in page.url or "unusual traffic" in page.locator("body").inner_text(timeout=5000).lower():
            return f"Google blocked the automated search for {query}. The browser is open, but results were not loaded; I will not claim the search succeeded."
        return f"Opened search results for {query}. Page title: {title}. Current page: {page.url}"

    def navigate(self, url: str) -> str:
        page = self._ensure_browser()
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        return f"Opened {page.url}. Page title: {page.title()}."

    def inspect_page(self) -> str:
        page = self._ensure_browser()
        title = page.title()
        url = page.url
        text = re.sub(r"\\s+", " ", page.locator("body").inner_text(timeout=10000)).strip()
        return f"Current page: {title} ({url})\\nVisible page text: {text[:6000]}"

    def click_images_tab(self) -> str:
        page = self._ensure_browser()
        # Use accessible DOM labels, not screen coordinates or a fixed location.
        candidates = page.get_by_role("link", name=re.compile(r"images", re.I))
        try:
            if candidates.count():
                candidates.first.click(timeout=5000)
                page.wait_for_load_state("domcontentloaded", timeout=15000)
                if "tbm=isch" in page.url or "udm=2" in page.url or "images" in page.url.lower():
                    return f"Opened the Images results tab. Current page: {page.url}"
        except Exception:
            pass
        # Search engines change their accessible labels frequently; use the query from
        # the current results URL and navigate directly to its image-results endpoint.
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(page.url)
        query = parse_qs(parsed.query).get("q", [""])[0]
        if query and "google." in parsed.netloc:
            page.goto("https://www.google.com/search?q=" + quote_plus(query) + "&udm=2",
                      wait_until="domcontentloaded", timeout=45000)
            if "/sorry/" not in page.url:
                return f"Opened Google Images for {query}. Current page: {page.url}"
        if query:
            page.goto("https://www.bing.com/images/search?q=" + quote_plus(query),
                      wait_until="domcontentloaded", timeout=45000)
            return f"Opened image results for {query}. Current page: {page.url}"
        return "I couldn't identify a search query from the current page, so I didn't change it."

    def download_visible_image(self, ordinal: int = 1) -> str:
        """Download a requested visible image by its rendered order (1-based)."""
        page = self._ensure_browser()
        images = page.locator("img")
        candidates = []
        for index in range(images.count()):
            image = images.nth(index)
            try:
                if not image.is_visible():
                    continue
                box = image.bounding_box()
                if not box or box["width"] < 80 or box["height"] < 80:
                    continue
                src = image.get_attribute("src") or image.get_attribute("data-src")
                if not src or src.startswith("data:image/svg"):
                    continue
                candidates.append((index, image))
            except Exception:
                continue
        if ordinal < 1 or ordinal > len(candidates):
            return f"Found {len(candidates)} downloadable visible images; image {ordinal} is unavailable, so nothing was downloaded."
        index, image = candidates[ordinal - 1]
        try:
            result = image.evaluate("""async (img) => {
                const src = img.currentSrc || img.src;
                const response = await fetch(src, {credentials: 'include'});
                if (!response.ok) throw new Error('Image fetch returned ' + response.status);
                const blob = await response.blob();
                return {name: (src.split('/').pop().split('?')[0] || 'page-image').slice(0,100),
                        type: blob.type || 'image/jpeg',
                        data: Array.from(new Uint8Array(await blob.arrayBuffer()))};
            }""")
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", result["name"]) or "page-image"
            if "." not in safe_name:
                safe_name += ".jpg"
            target = self.download_dir / safe_name
            if target.exists():
                target = self.download_dir / f"{target.stem}_{index}{target.suffix}"
            target.write_bytes(bytes(result["data"]))
            if target.stat().st_size == 0:
                target.unlink(missing_ok=True)
                return "The selected image returned an empty file; nothing was saved."
            return f"Downloaded image {ordinal} from the current page to {target}."
        except Exception as exc:
            return f"Could not download image {ordinal}: {exc}"

    def close(self) -> None:
        # Always attempt to stop the Node driver, even if Firefox/context shutdown
        # races with an already-closing window and raises a transport error.
        context, self._context = self._context, None
        playwright, self._playwright = self._playwright, None
        self._page = None
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass
