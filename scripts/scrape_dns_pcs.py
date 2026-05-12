"""Scrape pre-built PC catalogue from dns-shop.ru/custompc/user-pc/.

The listing page is protected by Qrator (JS challenge) and uses infinite
scroll. We use patchright (Playwright fork tuned against Qrator/Cloudflare)
in non-headless mode via WSLg / a real display — that's the only combo we
found that reliably passes the challenge.

Output: one JSON file with a flat list of builds:

    {
        "scraped_at": "2026-05-11T18:30:00+00:00",
        "source_url": "https://www.dns-shop.ru/custompc/user-pc/",
        "total_listed": 5076,
        "builds": [
            {
                "guid": "4bfaafc8-4dc5-...",
                "url": "https://www.dns-shop.ru/user-pc/configuration/4bfaafc84dc50140/",
                "price_rub": 57142,
                "cpu_slug": "amd-ryzen-5-5500-oem",
                "cpu_name": "AMD Ryzen 5 5500 OEM",
                "gpu_slug": "kfa2-geforce-rtx-3050-x-black-35nrldhp9odk",
                "gpu_name": "KFA2 GeForce RTX 3050 X Black",
                "ram_slug": "adata-premier-ad4u32008g22-dtgn-16-gb",
                "ram_gb": 16,
                "components": {
                    "motherboard": "gigabyte-a520m-k-v2",
                    "case":        "ardor-gaming-rare-nx30-lite-cernyj",
                    "psu":         "deepcool-pf550-r-pf550d-ha0b-eu-cernyj",
                    ...
                }
            },
            ...
        ]
    }

We only normalise CPU / GPU / RAM (the fields the predictor needs); everything
else is kept as a slug so downstream code can decide how to deal with it.

Usage:
    python scripts/scrape_dns_pcs.py                     # full catalogue
    python scripts/scrape_dns_pcs.py --max-builds 100    # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from patchright.async_api import async_playwright

LOG = logging.getLogger("scrape_dns_pcs")

LISTING_URL = "https://www.dns-shop.ru/custompc/user-pc/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "specs" / "dns_pcs.json"

# DNS uses transliterated Russian for component-type prefixes. Each maps to a
# normalised slot name in our JSON.
SLUG_PREFIX_TO_SLOT = {
    "processor-": "cpu",
    "videokarta-": "gpu",
    "operativnaa-pamat-": "ram",
    "materinskaa-plata-": "motherboard",
    "korpus-": "case",
    "blok-pitania-": "psu",
    "kuler-dla-processora-": "cooler",
    "ventilator-": "fan",
}

PRODUCT_HREF_RE = re.compile(r"^/product/[^/]+/([^/]+)/?$")
RAM_GB_RE = re.compile(r"(?<![\d])(\d+)\s*-?\s*gb(?:-|$)")


@dataclass
class Build:
    guid: str
    url: str
    price_rub: int | None
    cpu_slug: str | None = None
    cpu_name: str | None = None
    gpu_slug: str | None = None
    gpu_name: str | None = None
    ram_slug: str | None = None
    ram_gb: int | None = None
    components: dict[str, str] = field(default_factory=dict)


def _slug_to_name(slug: str) -> str:
    """Best-effort prettify of a dns-shop slug.

    `amd-ryzen-5-5500-oem` → `AMD Ryzen 5 5500 OEM`. Brand acronyms and a few
    well-known tokens get force-uppercased so the result is easier to fuzzy-
    match against our cpu/gpu spec catalogues.
    """
    parts = slug.split("-")
    upper = {"amd", "intel", "nvidia", "msi", "asus", "asrock", "gigabyte",
             "kfa2", "rtx", "gtx", "geforce", "radeon", "ddr3", "ddr4", "ddr5",
             "oem", "box", "wof", "ssd", "hdd", "nvme", "sata", "pcie",
             "ti", "xt", "super"}
    out = []
    for p in parts:
        if p.lower() in upper:
            out.append(p.upper())
        elif p.isdigit():
            out.append(p)
        else:
            out.append(p.capitalize())
    return " ".join(out)


def _classify_slug(slug: str) -> tuple[str | None, str]:
    """Return (slot_name, slug_without_prefix). slot_name is None if unknown."""
    for prefix, slot in SLUG_PREFIX_TO_SLOT.items():
        if slug.startswith(prefix):
            return slot, slug[len(prefix):]
    return None, slug


def _parse_price(text: str) -> int | None:
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else None


def parse_card(card_html: str) -> Build | None:
    soup = BeautifulSoup(card_html, "html.parser")
    root = soup.select_one("div.catalog-configuration")
    if root is None:
        return None
    guid = root.get("data-configuration-guid")
    if not guid:
        return None

    # Build self-link: /user-pc/configuration/<guid_no_dashes>/
    build_url = None
    for a in root.find_all("a", href=True):
        href = a["href"]
        if "/user-pc/configuration/" in href and "#" not in href:
            build_url = f"https://www.dns-shop.ru{href}"
            break

    price = None
    price_el = root.select_one(".product-buy__price")
    if price_el:
        price = _parse_price(price_el.get_text(strip=True))

    build = Build(guid=guid, url=build_url or "", price_rub=price)

    for a in root.select("a[href]"):
        m = PRODUCT_HREF_RE.match(a["href"])
        if not m:
            continue
        slug = m.group(1)
        slot, stripped = _classify_slug(slug)
        if slot is None:
            continue
        # First occurrence wins (DNS sometimes duplicates)
        if slot == "cpu" and build.cpu_slug is None:
            build.cpu_slug = stripped
            build.cpu_name = _slug_to_name(stripped)
        elif slot == "gpu" and build.gpu_slug is None:
            build.gpu_slug = stripped
            build.gpu_name = _slug_to_name(stripped)
        elif slot == "ram" and build.ram_slug is None:
            build.ram_slug = stripped
            ram_m = RAM_GB_RE.search(stripped)
            if ram_m:
                build.ram_gb = int(ram_m.group(1))
        else:
            build.components.setdefault(slot, stripped)
    return build


async def _wait_for_challenge(page, timeout_s: int = 45) -> None:
    """Block until the Qrator JS challenge clears (page body sized + no qauth)."""
    for i in range(timeout_s):
        html = await page.content()
        if "qauth" not in html and len(html) > 50_000:
            LOG.info("challenge cleared after %ds", i)
            return
        await page.wait_for_timeout(1_000)
    raise RuntimeError(f"Qrator challenge did not clear within {timeout_s}s")


async def _read_total_listed(page) -> int | None:
    el = page.locator(".products-count").first
    if await el.count() == 0:
        return None
    txt = await el.inner_text()
    return _parse_price(txt)  # extracts digits


async def _hydrate(page, target: int, max_rounds: int = 20) -> tuple[int, int]:
    """Run scrollIntoView rounds until enough cards are priced or progress stalls.

    Returns (priced_count, total_count) at the end.
    """
    prev_priced = -1
    for round_ in range(max_rounds):
        await page.evaluate(
            """
            async () => {
                const cards = document.querySelectorAll("div.catalog-configuration");
                for (let i = 0; i < cards.length; i++) {
                    if (!cards[i].querySelector(".product-buy__price")) {
                        cards[i].scrollIntoView({block: "center"});
                        await new Promise(r => setTimeout(r, 150));
                    }
                }
            }
            """,
        )
        await page.wait_for_timeout(1_200)
        total = await page.locator("div.catalog-configuration").count()
        priced = await page.locator(
            "div.catalog-configuration:has(.product-buy__price)"
        ).count()
        LOG.info("  hydration round %d: priced=%d / total=%d",
                 round_ + 1, priced, total)
        if priced >= target or priced == prev_priced:
            return priced, total
        prev_priced = priced
    return priced, total


async def _collect_builds(page, max_builds: int | None) -> list[Build]:
    cards_html = await page.locator("div.catalog-configuration").evaluate_all(
        "els => els.map(e => e.outerHTML)"
    )
    builds: list[Build] = []
    skipped = 0
    for h in cards_html:
        b = parse_card(h)
        if b is None or b.price_rub is None:
            skipped += 1
            continue
        builds.append(b)
        if max_builds is not None and len(builds) >= max_builds:
            break
    LOG.info("collected %d priced builds (skipped %d without price)", len(builds), skipped)
    return builds


def _save_snapshot(builds: list[Build], total_listed: int | None, output: Path,
                   partial: bool) -> None:
    payload = {
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": LISTING_URL,
        "total_listed": total_listed,
        "scraped_count": len(builds),
        "partial": partial,
        "builds": [asdict(b) for b in builds],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output)
    LOG.info("%s snapshot -> %s (%d builds)",
             "partial" if partial else "final", output, len(builds))


async def scrape(max_builds: int | None,
                 scroll_delay_ms: int,
                 snapshot_every: int,
                 user_data_dir: Path,
                 output: Path) -> dict:
    user_data_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=False,
            channel="chromium",
            no_viewport=True,
        )
        page = await ctx.new_page()
        LOG.info("goto %s", LISTING_URL)
        await page.goto(LISTING_URL, wait_until="domcontentloaded", timeout=60_000)
        await _wait_for_challenge(page)
        await page.wait_for_selector("div.catalog-configuration", timeout=30_000)
        total_listed = await _read_total_listed(page)
        LOG.info("total listed: %s", total_listed)

        target = min(total_listed or 10_000, max_builds or 10_000)
        builds: list[Build] = []
        last_snapshot_at = 0
        last_count = 0
        stalls = 0

        async def checkpoint(reason: str) -> None:
            nonlocal builds
            LOG.info("checkpoint (%s) — hydrating + saving partial", reason)
            await _hydrate(page, target)
            builds = await _collect_builds(page, max_builds)
            _save_snapshot(builds, total_listed, output, partial=True)

        try:
            while True:
                n = await page.locator("div.catalog-configuration").count()
                LOG.info("cards loaded: %d / %d", n, target)
                if n - last_snapshot_at >= snapshot_every and n > 0:
                    await checkpoint(f"every {snapshot_every}")
                    last_snapshot_at = n
                    if len(builds) >= target:
                        break
                if n >= target:
                    break
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(scroll_delay_ms)
                n_new = await page.locator("div.catalog-configuration").count()
                if n_new == last_count:
                    stalls += 1
                    if stalls >= 5:
                        LOG.warning("scroll stalled at %d cards — stopping", n_new)
                        break
                else:
                    stalls = 0
                last_count = n_new
        except (KeyboardInterrupt, Exception) as exc:
            LOG.warning("interrupted (%s) — saving partial before exit", type(exc).__name__)
            try:
                await checkpoint("error")
            except Exception as inner:
                LOG.error("checkpoint after error failed: %s", inner)
            raise
        finally:
            # Final hydrate + save regardless of how we exited the loop.
            try:
                await _hydrate(page, target)
                builds = await _collect_builds(page, max_builds)
                _save_snapshot(builds, total_listed, output, partial=False)
            except Exception as e:
                LOG.error("final save failed: %s", e)
            await ctx.close()

    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": LISTING_URL,
        "total_listed": total_listed,
        "scraped_count": len(builds),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-builds", type=int, default=None,
                   help="cap on the number of builds to collect (default: all)")
    p.add_argument("--scroll-delay-ms", type=int, default=1500,
                   help="wait between scrolls (default: 1500)")
    p.add_argument("--snapshot-every", type=int, default=300,
                   help="hydrate + save partial JSON every N cards (default: 300)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT,
                   help=f"output JSON path (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})")
    p.add_argument("--user-data-dir", type=Path,
                   default=PROJECT_ROOT / "tmp" / "patchright-userdata",
                   help="persistent Chromium profile dir")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(levelname)s] %(message)s")
    result = asyncio.run(scrape(args.max_builds, args.scroll_delay_ms,
                                args.snapshot_every, args.user_data_dir,
                                args.output))
    LOG.info("done: %d builds (target was %s)",
             result["scraped_count"], args.max_builds or result["total_listed"])


if __name__ == "__main__":
    main()
