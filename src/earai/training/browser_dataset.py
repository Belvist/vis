"""Browser-based UI Dataset Generator - uses Playwright to render HTML/CSS and extract DOM ground truth"""
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from urllib.parse import urlparse

# UI class mapping (15 classes)
UI_CLASSES = [
    'navbar', 'hero', 'section', 'container', 'card', 'button', 'input',
    'image', 'icon', 'heading', 'paragraph', 'badge', 'modal', 'footer', 'other'
]
UI_CLASS_TO_IDX = {c: i for i, c in enumerate(UI_CLASSES)}


def extract_domain(url: str) -> str:
    """Extract domain from URL for train/val split"""
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    return domain


async def extract_dom_elements(page, viewport_width: int, viewport_height: int) -> List[Dict]:
    """Extract all visible UI elements from the DOM with computed styles and bounding boxes"""

    js_script = """
    () => {
        const elements = [];
        const allElements = document.querySelectorAll('*');

        for (const el of allElements) {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;

            const rect = el.getBoundingClientRect();
            if (rect.width <= 1 || rect.height <= 1) continue;
            if (rect.x + rect.width < 0 || rect.y + rect.height < 0) continue;
            if (rect.x > window.innerWidth || rect.y > window.innerHeight) continue;

            let text = '';
            for (const node of el.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    const t = node.textContent.trim();
                    if (t) text += t.replace(/\\0/g, '');
                }
            }
            text = text.slice(0, 200);

            let uiType = 'other';
            const tag = el.tagName.toLowerCase();
            const className = (el.className && el.className.baseVal !== undefined) ? el.className.baseVal : (el.className || '');
            const id = el.id || '';
            const role = el.getAttribute('role') || '';

            if (tag === 'nav' || className.includes('nav') || role === 'navigation') uiType = 'navbar';
            else if (tag === 'header' || className.includes('hero') || className.includes('banner')) uiType = 'hero';
            else if (tag === 'section' || className.includes('section')) uiType = 'section';
            else if (tag === 'main' || tag === 'div' && (className.includes('container') || className.includes('wrapper'))) uiType = 'container';
            else if (tag === 'article' || className.includes('card') || className.includes('tile')) uiType = 'card';
            else if (tag === 'button' || tag === 'a' && (className.includes('btn') || className.includes('button')) || role === 'button') uiType = 'button';
            else if (tag === 'input' || tag === 'textarea' || tag === 'select') uiType = 'input';
            else if (tag === 'img' || tag === 'picture' || className.includes('image') || className.includes('img-')) uiType = 'image';
            else if (tag === 'svg' || tag === 'i' || className.includes('icon')) uiType = 'icon';
            else if (['h1','h2','h3','h4','h5','h6'].includes(tag)) uiType = 'heading';
            else if (tag === 'p' || tag === 'span' || tag === 'div' && text.length > 20) uiType = 'paragraph';
            else if (className.includes('badge') || className.includes('tag') || className.includes('label')) uiType = 'badge';
            else if (tag === 'dialog' || className.includes('modal') || role === 'dialog') uiType = 'modal';
            else if (tag === 'footer') uiType = 'footer';

            function sanitize(str) { return str ? str.replace(/\\0/g, '') : ''; }
            const bgColor = sanitize(style.backgroundColor);
            const color = sanitize(style.color);
            const borderRadius = sanitize(style.borderRadius);
            const fontSize = sanitize(style.fontSize);
            const fontWeight = sanitize(style.fontWeight);
            const lineHeight = sanitize(style.lineHeight);

            function parseColor(cssColor) {
                if (!cssColor || cssColor === 'rgba(0, 0, 0, 0)' || cssColor === 'transparent') return [0.5, 0.5, 0.5];
                const match = cssColor.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                if (match) return [parseInt(match[1])/255, parseInt(match[2])/255, parseInt(match[3])/255];
                return [0.5, 0.5, 0.5];
            }

            function parsePx(val) { if (!val) return 0; const m = val.match(/([\\d.]+)px/); return m ? parseFloat(m[1]) : 0; }

            let parentId = null;
            if (el.parentElement) {
                const parentTag = sanitize(el.parentElement.tagName.toLowerCase());
                const parentIdStr = sanitize(el.parentElement.id || '');
                const parentClass = sanitize(Array.from(el.parentElement.classList).join('_') || 'anon');
                parentId = parentTag + '_' + (parentIdStr || parentClass);
            }

            elements.push({
                type: uiType,
                bbox: [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height],
                text: text,
                parent_id: parentId,
                style: {
                    background: parseColor(bgColor),
                    foreground: parseColor(color),
                    radius: parsePx(borderRadius),
                    font_size: parsePx(fontSize),
                    font_weight: parseInt(fontWeight) || 400,
                    line_height: parsePx(lineHeight) || parsePx(fontSize) * 1.5
                }
            });
        }
        return elements;
    }
    """

    elements = await page.evaluate(js_script)
    return elements


async def generate_dataset_from_urls(urls: List[str], output_dir: str, viewport_sizes: List[tuple] = None, split_name: str = 'train'):
    """Generate dataset from a list of URLs using Playwright"""
    from playwright.async_api import async_playwright

    if viewport_sizes is None:
        viewport_sizes = [(1440, 900), (1920, 1080), (1366, 768), (375, 667), (768, 1024)]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / 'images'
    images_dir.mkdir(exist_ok=True)

    all_samples = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for url_idx, url in enumerate(tqdm(urls, desc=f"Processing {split_name} URLs")):
            for vp_idx, (vw, vh) in enumerate(viewport_sizes):
                context = await browser.new_context(viewport={'width': vw, 'height': vh}, device_scale_factor=1)
                page = await context.new_page()

                try:
                    await page.goto(url, wait_until='networkidle', timeout=30000)
                    await page.wait_for_load_state('networkidle')
                    await asyncio.sleep(1)

                    elements = await extract_dom_elements(page, vw, vh)

                    if not elements:
                        print(f"  No elements found for {url} @ {vw}x{vh}")
                        await context.close()
                        continue

                    screenshot_bytes = await page.screenshot(full_page=False)
                    import io
                    screenshot = Image.open(io.BytesIO(screenshot_bytes))
                    screenshot_np = np.array(screenshot)

                    if screenshot_np.shape[2] == 4:
                        screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGBA2RGB)

                    domain = extract_domain(url).replace('.', '_')
                    img_name = f"{domain}_{url_idx}_vp{vp_idx}.png"
                    img_path = images_dir / img_name
                    Image.fromarray(screenshot_np).save(img_path)

                    ui_elements = []
                    for i, elem in enumerate(elements):
                        bbox = elem['bbox']
                        norm_bbox = [bbox[0]/vw, bbox[1]/vh, bbox[2]/vw, bbox[3]/vh]
                        norm_bbox = [max(0, min(1, x)) for x in norm_bbox]

                        class_id = UI_CLASS_TO_IDX.get(elem['type'], UI_CLASS_TO_IDX['other'])

                        def clean_str(s):
                            if isinstance(s, str): return s.replace('\x00', '')
                            return s

                        ui_elements.append({
                            'bbox': norm_bbox,
                            'class_id': class_id,
                            'class_name': clean_str(elem['type']),
                            'text': clean_str(elem.get('text', '')),
                            'parent_id': clean_str(elem.get('parent_id', '')) if elem.get('parent_id') else None,
                            'style': elem['style']
                        })

                    sample = {
                        'image_id': f"{domain}_{url_idx}_vp{vp_idx}",
                        'image_path': str(img_path),
                        'viewport': [vw, vh],
                        'domain': domain,
                        'url': url,
                        'ui_elements': ui_elements,
                        'source': 'browser'
                    }
                    all_samples.append(sample)
                    print(f"  {url} @ {vw}x{vh}: {len(ui_elements)} elements")

                except Exception as e:
                    import traceback
                    print(f"  Error processing {url} @ {vw}x{vh}: {e}")
                    traceback.print_exc()
                finally:
                    await context.close()

        await browser.close()

    manifest_path = output_path / f'manifest_{split_name}.json'
    with open(manifest_path, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nGenerated {len(all_samples)} {split_name} samples")
    print(f"Manifest saved to {manifest_path}")
    return all_samples


def create_web_ui_dataset(config: dict, force_regenerate: bool = False) -> List[Dict]:
    """Create or load web UI dataset from browser-rendered pages with domain-based split"""
    data_root = Path(config.get('data_root', 'data/web_ui'))
    manifest_path = data_root / 'manifest.json'

    train_urls = config.get('train_urls', [])
    val_urls = config.get('val_urls', [])

    train_domains = [extract_domain(u) for u in train_urls]
    val_domains = [extract_domain(u) for u in val_urls]

    if not force_regenerate and manifest_path.exists():
        print(f"Loading existing dataset from {manifest_path}")
        with open(manifest_path) as f:
            return json.load(f)

    if not train_urls or not val_urls:
        urls = config.get('urls', [
            'https://stripe.com', 'https://linear.app', 'https://vercel.com',
            'https://github.com', 'https://tailwindcss.com', 'https://react.dev',
            'https://nextjs.org', 'https://figma.com', 'https://notion.so', 'https://airbnb.com',
        ])
        viewport_sizes = config.get('viewport_sizes', [(1440, 900), (1920, 1080), (1366, 768), (375, 667), (768, 1024)])
        print(f"Generating dataset from {len(urls)} URLs x {len(viewport_sizes)} viewports")
        return asyncio.run(generate_dataset_from_urls(urls, str(data_root), viewport_sizes))

    viewport_sizes = config.get('viewport_sizes', [(1440, 900), (1920, 1080), (1366, 768), (375, 667), (768, 1024)])

    print(f"Generating TRAIN dataset from {len(train_urls)} URLs...")
    train_samples = asyncio.run(generate_dataset_from_urls(train_urls, str(data_root), viewport_sizes, 'train'))

    print(f"Generating VAL dataset from {len(val_urls)} URLs...")
    val_samples = asyncio.run(generate_dataset_from_urls(val_urls, str(data_root), viewport_sizes, 'val'))

    all_samples = train_samples + val_samples

    with open(manifest_path, 'w') as f:
        json.dump(all_samples, f, indent=2)

    split_info = {'train_domains': train_domains, 'val_domains': val_domains, 'train_count': len(train_samples), 'val_count': len(val_samples)}
    with open(data_root / 'domain_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nTotal: {len(all_samples)} samples (train: {len(train_samples)}, val: {len(val_samples)})")
    return all_samples


if __name__ == '__main__':
    import yaml
    with open('configs/gate2.yaml') as f:
        config = yaml.safe_load(f)
    create_web_ui_dataset(config, force_regenerate=True)
