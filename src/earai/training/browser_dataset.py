"""Browser-rendered Web UI dataset with DOM/CSS ground truth."""
import asyncio
import io
import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
from PIL import Image
from tqdm import tqdm

UI_CLASSES = [
    "navbar", "hero", "section", "container", "card", "button", "input",
    "image", "icon", "heading", "paragraph", "badge", "modal", "footer", "link",
]
UI_CLASS_TO_IDX = {name: i for i, name in enumerate(UI_CLASSES)}


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


async def extract_dom_elements(page) -> List[Dict]:
    """Return visible semantic UI elements in viewport coordinates."""
    script = r"""
    () => {
      const isVisible = (el) => {
        const s = getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity || '1') <= 0) return false;
        const r = el.getBoundingClientRect();
        return r.width > 2 && r.height > 2 &&
               r.right > 0 && r.bottom > 0 &&
               r.left < innerWidth && r.top < innerHeight;
      };

      const classify = (el) => {
        const tag = el.tagName.toLowerCase();
        const cls = String(el.className && el.className.baseVal !== undefined ? el.className.baseVal : (el.className || '')).toLowerCase();
        const role = (el.getAttribute('role') || '').toLowerCase();
        const directText = Array.from(el.childNodes)
          .filter(n => n.nodeType === Node.TEXT_NODE)
          .map(n => (n.textContent || '').trim()).join(' ').trim();

        if (tag === 'nav' || role === 'navigation' || cls.includes('navbar') || cls.includes('nav-')) return 'navbar';
        if (tag === 'header' || cls.includes('hero') || cls.includes('banner')) return 'hero';
        if (tag === 'dialog' || role === 'dialog' || cls.includes('modal')) return 'modal';
        if (tag === 'footer') return 'footer';
        if (tag === 'button' || role === 'button' || (tag === 'a' && (cls.includes('btn') || cls.includes('button')))) return 'button';
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return 'input';
        if (tag === 'img' || tag === 'picture' || cls.includes('image') || cls.includes('img-')) return 'image';
        if (tag === 'svg' || tag === 'i' || cls.includes('icon')) return 'icon';
        if (/^h[1-6]$/.test(tag)) return 'heading';
        if (tag === 'section' || cls.includes('section')) return 'section';
        if (tag === 'article' || cls.includes('card') || cls.includes('tile')) return 'card';
        if (cls.includes('badge') || cls.includes('pill') || cls.includes('tag')) return 'badge';
        if (tag === 'a') return 'link';
        if (tag === 'main' || (tag === 'div' && (cls.includes('container') || cls.includes('wrapper')))) return 'container';
        if (tag === 'p' || (tag === 'span' && directText.length > 0) || (tag === 'div' && directText.length > 20)) return 'paragraph';
        return null;
      };

      const parseColor = (value, el, prop) => {
        let v = value;
        if (!v || v === 'transparent' || v === 'rgba(0, 0, 0, 0)') {
          let p = el.parentElement;
          while (p) {
            const pv = getComputedStyle(p)[prop];
            if (pv && pv !== 'transparent' && pv !== 'rgba(0, 0, 0, 0)') { v = pv; break; }
            p = p.parentElement;
          }
        }
        const m = String(v || '').match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        return m ? [Number(m[1])/255, Number(m[2])/255, Number(m[3])/255] : [0.5, 0.5, 0.5];
      };
      const px = (v) => {
        const m = String(v || '').match(/([\d.]+)px/);
        return m ? Number(m[1]) : 0;
      };

      const semantic = Array.from(document.querySelectorAll('*'))
        .filter(isVisible)
        .map(el => ({el, type: classify(el)}))
        .filter(x => x.type !== null);

      const idMap = new Map();
      semantic.forEach((x, i) => idMap.set(x.el, 'e' + i));

      return semantic.map(({el, type}, i) => {
        const s = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        let parent = el.parentElement;
        while (parent && !idMap.has(parent)) parent = parent.parentElement;
        const text = (el.innerText || '').replace(/\0/g, '').trim().replace(/\s+/g, ' ').slice(0, 240);
        return {
          element_id: 'e' + i,
          parent_id: parent ? idMap.get(parent) : null,
          type,
          bbox: [r.left, r.top, r.right, r.bottom],
          text,
          style: {
            background: parseColor(s.backgroundColor, el, 'backgroundColor'),
            foreground: parseColor(s.color, el, 'color'),
            radius: px(s.borderRadius),
            font_size: px(s.fontSize),
            font_weight: Number.parseInt(s.fontWeight) || 400,
            line_height: px(s.lineHeight) || (px(s.fontSize) * 1.5),
          }
        };
      });
    }
    """
    return await page.evaluate(script)


async def _capture_url(page, url: str, viewport: tuple, scroll_fractions: List[float],
                       images_dir: Path, url_idx: int, split_name: str) -> List[Dict]:
    vw, vh = viewport
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(1200)

    scroll_height = await page.evaluate("Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
    max_scroll = max(0, int(scroll_height) - vh)
    scroll_positions = sorted({int(max_scroll * max(0.0, min(1.0, float(f)))) for f in scroll_fractions})

    domain = extract_domain(url)
    safe_domain = domain.replace(".", "_")
    samples = []

    for scroll_idx, scroll_y in enumerate(scroll_positions):
        await page.evaluate("(y) => window.scrollTo(0, y)", scroll_y)
        await page.wait_for_timeout(200)

        elements = await extract_dom_elements(page)
        if not elements:
            continue

        shot = await page.screenshot(full_page=False)
        image = Image.open(io.BytesIO(shot)).convert("RGB")
        image_name = f"{split_name}_{safe_domain}_{url_idx}_vp{vw}x{vh}_s{scroll_idx}.png"
        image_path = images_dir / image_name
        image.save(image_path)

        ui_elements = []
        for elem in elements:
            x1, y1, x2, y2 = elem["bbox"]
            norm = [
                max(0.0, min(1.0, x1 / vw)),
                max(0.0, min(1.0, y1 / vh)),
                max(0.0, min(1.0, x2 / vw)),
                max(0.0, min(1.0, y2 / vh)),
            ]
            if norm[2] - norm[0] <= 0.002 or norm[3] - norm[1] <= 0.002:
                continue
            ui_elements.append({
                "element_id": elem["element_id"],
                "parent_id": elem.get("parent_id"),
                "bbox": norm,
                "class_id": UI_CLASS_TO_IDX[elem["type"]],
                "class_name": elem["type"],
                "text": elem.get("text", ""),
                "style": elem["style"],
            })

        if not ui_elements:
            continue

        samples.append({
            "image_id": image_name[:-4],
            "image_path": str(image_path),
            "viewport": [vw, vh],
            "scroll_y": scroll_y,
            "domain": domain,
            "url": url,
            "split": split_name,
            "ui_elements": ui_elements,
            "source": "browser",
        })
    return samples


async def generate_dataset_from_urls(urls: List[str], output_dir: str,
                                     viewport_sizes: List[tuple],
                                     scroll_fractions: List[float],
                                     split_name: str) -> List[Dict]:
    from playwright.async_api import async_playwright

    root = Path(output_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for url_idx, url in enumerate(tqdm(urls, desc=f"Rendering {split_name}")):
            for viewport in viewport_sizes:
                vw, vh = int(viewport[0]), int(viewport[1])
                context = await browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=1)
                page = await context.new_page()
                try:
                    samples.extend(await _capture_url(
                        page, url, (vw, vh), scroll_fractions, images_dir, url_idx, split_name
                    ))
                except Exception as exc:
                    print(f"[Gate2] skip {url} {vw}x{vh}: {exc}")
                finally:
                    await context.close()
        await browser.close()

    with open(root / f"manifest_{split_name}.json", "w") as f:
        json.dump(samples, f, indent=2)
    return samples


def _load_manifest(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def _manifest_valid(samples: List[Dict], split: str, allowed_domains: set) -> bool:
    if not samples:
        return False
    for sample in samples:
        if sample.get("split") != split:
            return False
        if sample.get("domain") not in allowed_domains:
            return False
        elems = sample.get("ui_elements")
        if not isinstance(elems, list) or not elems:
            return False
        first = elems[0]
        if "element_id" not in first or "parent_id" not in first:
            return False
    return True


def create_web_ui_dataset(config: dict, force_regenerate: bool = False) -> List[Dict]:
    """Create/load strict domain-separated browser dataset."""
    root = Path(config.get("data_root", "./data/web_ui"))
    root.mkdir(parents=True, exist_ok=True)
    train_manifest = root / "manifest_train.json"
    val_manifest = root / "manifest_val.json"

    train_urls = list(config.get("train_urls", []))
    val_urls = list(config.get("val_urls", []))
    if not train_urls or not val_urls:
        raise RuntimeError("Gate 2 requires non-empty train_urls and val_urls")

    train_domains = sorted({extract_domain(u) for u in train_urls})
    val_domains = sorted({extract_domain(u) for u in val_urls})
    overlap = sorted(set(train_domains) & set(val_domains))
    if overlap:
        raise RuntimeError(f"Gate 2 domain leakage: {overlap}")

    viewports = [tuple(v) for v in config.get("viewport_sizes", [(1440, 900)])]
    scroll_fractions = list(config.get("scroll_fractions", [0.0]))

    train_allowed = set(train_domains)
    val_allowed = set(val_domains)

    train = _load_manifest(train_manifest) if train_manifest.exists() else []
    if force_regenerate or not _manifest_valid(train, "train", train_allowed):
        train = asyncio.run(generate_dataset_from_urls(
            train_urls, str(root), viewports, scroll_fractions, "train"
        ))

    val = _load_manifest(val_manifest) if val_manifest.exists() else []
    if force_regenerate or not _manifest_valid(val, "val", val_allowed):
        val = asyncio.run(generate_dataset_from_urls(
            val_urls, str(root), viewports, scroll_fractions, "val"
        ))

    with open(root / "train_domains.json", "w") as f:
        json.dump(train_domains, f, indent=2)
    with open(root / "val_domains.json", "w") as f:
        json.dump(val_domains, f, indent=2)
    with open(root / "domain_split.json", "w") as f:
        json.dump({
            "train_domains": train_domains,
            "val_domains": val_domains,
            "train_count": len(train),
            "val_count": len(val),
        }, f, indent=2)

    return train + val


if __name__ == "__main__":
    import yaml
    with open("configs/gate2.yaml") as f:
        cfg = yaml.safe_load(f)
    data = create_web_ui_dataset(cfg, force_regenerate=True)
    print(f"Generated {len(data)} screenshots")
