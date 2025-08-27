
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast eoPortal scraper with JavaScript rendering and structured content extraction
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Install with: pip install selenium")

# Base URLs
MISSION_BASE = "https://www.eoportal.org/satellite-missions"
SITEMAP_URL = "https://www.eoportal.org/sitemap.xml"

DEFAULT_UA = (
    "Mozilla/5.0 (compatible; MESA-RAG-eoportal-scraper/3.0; +https://example.org; research/academic use) "
    "requests/" + requests.__version__
)

def sha256(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def clean_text_from_html(html: str) -> str:
    """Extract clean text from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

def extract_structured_content(html: str) -> dict:
    """Extract structured content including tables and images"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract tables
    tables = []
    for i, table in enumerate(soup.find_all("table")):
        table_data = {
            "table_id": i,
            "rows": []
        }
        
        for row in table.find_all("tr"):
            cells = []
            for cell in row.find_all(["td", "th"]):
                cells.append(cell.get_text(strip=True))
            if cells:  # Only add non-empty rows
                table_data["rows"].append(cells)
        
        if table_data["rows"]:  # Only add tables with content
            tables.append(table_data)
    
    # Extract images
    images = []
    for i, img in enumerate(soup.find_all("img")):
        img_data = {
            "image_id": i,
            "src": img.get("src", ""),
            "alt": img.get("alt", ""),
            "title": img.get("title", ""),
            "caption": ""
        }
        
        # Try to find caption in nearby elements
        parent = img.parent
        if parent:
            caption_elem = parent.find("figcaption") or parent.find_next("p")
            if caption_elem:
                img_data["caption"] = caption_elem.get_text(strip=True)
        
        images.append(img_data)
    
    return {
        "tables": tables,
        "images": images
    }

def get_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.text:
        return soup.title.text.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""

def make_session(user_agent: str, timeout: int = 30) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    s.request_timeout = timeout
    return s

@retry(wait=wait_exponential_jitter(initial=1, max=20), stop=stop_after_attempt(5))
def fetch(session: requests.Session, url: str) -> requests.Response:
    resp = session.get(url, timeout=session.request_timeout)
    resp.raise_for_status()
    return resp

def setup_selenium_driver(cookie_file: Path = None):
    """Setup Chrome driver for JavaScript rendering"""
    if not SELENIUM_AVAILABLE:
        return None
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument(f'--user-agent={DEFAULT_UA}')
    
    # Add preferences to handle notifications and popups
    prefs = {
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0
    }
    options.add_experimental_option("prefs", prefs)
    
    try:
        driver = webdriver.Chrome(options=options)
        
        # Load cookies if available
        if cookie_file and cookie_file.exists():
            driver.get("https://www.eoportal.org")
            with open(cookie_file, 'r') as f:
                cookies = json.load(f)
                for cookie in cookies:
                    try:
                        driver.add_cookie(cookie)
                    except:
                        pass
        
        return driver
    except Exception as e:
        print(f"Could not setup Chrome driver: {e}")
        return None

def save_cookies(driver, cookie_file: Path):
    """Save cookies from Selenium driver to file"""
    try:
        cookies = driver.get_cookies()
        with open(cookie_file, 'w') as f:
            json.dump(cookies, f)
    except Exception as e:
        print(f"Failed to save cookies: {e}")

def handle_cookie_consent(driver, cookie_file: Path = None):
    """Handle cookie consent popup if present"""
    try:
        # Common cookie consent button patterns
        consent_patterns = [
            "//button[contains(text(), 'Accept')]",
            "//button[contains(text(), 'Accept all')]",
            "//button[contains(text(), 'Accept All')]",
            "//button[contains(text(), 'I agree')]",
            "//button[contains(text(), 'OK')]",
            "//button[contains(@class, 'accept')]",
            "//button[contains(@class, 'consent')]",
            "//a[contains(text(), 'Accept')]",
            "//a[contains(@class, 'accept')]",
            "//div[contains(@class, 'cookie')]//button",
            "//div[contains(@id, 'cookie')]//button",
            "//button[@id='accept-cookies']",
            "//button[@class='cookie-consent-accept']"
        ]
        
        for pattern in consent_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        driver.execute_script("arguments[0].click();", element)
                        time.sleep(1)
                        # Save cookies after accepting
                        if cookie_file:
                            save_cookies(driver, cookie_file)
                        return True
            except:
                continue
                
        # Try to find any visible button with cookie-related text
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            text = button.text.lower()
            if any(word in text for word in ['accept', 'agree', 'ok', 'consent', 'allow']) and button.is_displayed():
                driver.execute_script("arguments[0].click();", button)
                time.sleep(1)
                # Save cookies after accepting
                if cookie_file:
                    save_cookies(driver, cookie_file)
                return True
                
    except Exception as e:
        print(f"Cookie consent handling failed: {e}")
    
    return False

def fetch_with_js(driver, url: str, cookie_file: Path = None, timeout: int = 30) -> tuple[str, str]:
    """Fetch page content with JavaScript rendering
    Returns: (page_source, final_url)
    """
    driver.get(url)
    
    # First check for cookie consent
    time.sleep(1)  # Give time for cookie popup to appear
    handle_cookie_consent(driver, cookie_file)
    
    # Wait for content to load (looking for actual content, not skeleton)
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: len(d.page_source) > 100000 or 
                     "eoPortal" in d.title or
                     len(d.find_elements(By.TAG_NAME, "p")) > 5
        )
        
        # Additional wait for dynamic content
        time.sleep(2)
        
    except Exception:
        print(f"Timeout waiting for content to load: {url}")
    
    # Get the final URL after any redirects
    final_url = driver.current_url
    
    return driver.page_source, final_url

def discover_from_sitemap(session: requests.Session) -> set[str]:
    """Discover mission URLs from sitemap (handles sitemap index)"""
    try:
        resp = fetch(session, SITEMAP_URL)
        root = ET.fromstring(resp.content)
        
        urls = set()
        
        # Check if this is a sitemap index
        if root.tag == "{http://www.sitemaps.org/schemas/sitemap/0.9}sitemapindex":
            print("Found sitemap index, fetching individual sitemaps...")
            sitemap_urls = []
            for sitemap_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
                loc_elem = sitemap_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc_elem is not None and loc_elem.text:
                    sitemap_urls.append(loc_elem.text)
            
            print(f"Found {len(sitemap_urls)} sitemaps to process")
            
            # Fetch each sitemap
            for sitemap_url in sitemap_urls:
                try:
                    print(f"Fetching {sitemap_url}")
                    resp = fetch(session, sitemap_url)
                    sitemap_root = ET.fromstring(resp.content)
                    
                    for url_elem in sitemap_root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                        loc_elem = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                        if loc_elem is not None and loc_elem.text:
                            url = loc_elem.text
                            if "/satellite-missions/" in url and not url.endswith("/satellite-missions/"):
                                urls.add(url)
                                
                except Exception as e:
                    print(f"Failed to fetch sitemap {sitemap_url}: {e}")
                    continue
        else:
            # Regular sitemap
            for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                loc_elem = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc_elem is not None and loc_elem.text:
                    url = loc_elem.text
                    if "/satellite-missions/" in url and not url.endswith("/satellite-missions/"):
                        urls.add(url)
        
        print(f"✓ Discovered {len(urls)} URLs from sitemap")
        return urls
        
    except Exception as e:
        print(f"Failed to fetch sitemap: {e}")
        return set()

def has_cookie_consent_page(html: str) -> bool:
    """Check if the page is showing a cookie consent page instead of content"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Check for common cookie consent indicators
    cookie_indicators = [
        "Cookie Notice",
        "Privacy Notice", 
        "Terms and Conditions",
        "Leave Feedback",
        "Contact",
        "About"
    ]
    
    text = soup.get_text()
    
    # If the page is very short and contains these elements, it's likely cookie consent
    if len(text.strip()) < 500:
        cookie_count = sum(1 for indicator in cookie_indicators if indicator in text)
        if cookie_count >= 3:
            return True
    
    # Look for specific cookie consent patterns
    if "cookie" in text.lower() and "consent" in text.lower():
        return True
        
    # Check if page only has navigation elements (typical of consent pages)
    if all(indicator in text for indicator in ["eoPortal", "Satellite Missions", "Search"]) and len(text.strip()) < 200:
        return True
    
    # Check for cookie consent overlays or modals
    cookie_elements = soup.find_all(attrs={"class": lambda x: x and any(word in x.lower() for word in ["cookie", "consent", "gdpr", "privacy"])})
    if cookie_elements:
        return True
    
    # Check for specific eoPortal patterns
    main_content = soup.find("main") or soup.find("article") or soup.find(attrs={"id": "content"})
    if not main_content or len(main_content.get_text(strip=True)) < 100:
        return True
        
    return False

def accept_cookies(session: requests.Session, base_url: str) -> bool:
    """Attempt to accept cookies by visiting the main page and setting consent"""
    try:
        # Visit main page first
        main_page = session.get("https://www.eoportal.org/")
        
        # Look for cookie acceptance endpoints or forms
        soup = BeautifulSoup(main_page.content, "html.parser")
        
        # Try common cookie consent approaches
        consent_urls = [
            "https://www.eoportal.org/cookie-consent",
            "https://www.eoportal.org/accept-cookies",
            "https://www.eoportal.org/privacy-accept"
        ]
        
        for url in consent_urls:
            try:
                response = session.post(url, data={"consent": "accept", "accept": "true"})
                if response.status_code < 400:
                    break
            except:
                continue
                
        # Set common cookie consent cookies manually
        session.cookies.set('cookie_consent', 'accepted', domain='eoportal.org')
        session.cookies.set('cookies_accepted', 'true', domain='eoportal.org')
        session.cookies.set('gdpr_consent', 'true', domain='eoportal.org')
        session.cookies.set('privacy_policy_accepted', 'true', domain='eoportal.org')
        
        return True
        
    except Exception as e:
        print(f"Cookie acceptance failed: {e}")
        return False

def download_mission_fast(session: requests.Session, driver, url: str, output_dir: Path, cookie_file: Path = None, use_js: bool = True, thread_driver: bool = False) -> dict:
    """Download mission with fast content extraction"""
    original_mission_name = url.split("/")[-1]
    mission_name = original_mission_name  # Default to original
    
    html_path = output_dir / "raw_html" / "missions" / f"{mission_name}.html"
    text_path = output_dir / "text" / "missions" / f"{mission_name}.txt"
    tables_path = output_dir / "structured" / "missions" / f"{mission_name}_tables.json"
    images_path = output_dir / "structured" / "missions" / f"{mission_name}_images.json"
    
    ensure_parent(html_path)
    ensure_parent(text_path)
    ensure_parent(tables_path)
    ensure_parent(images_path)
    
    # Create a thread-local driver if needed
    local_driver = None
    try:
        # Try JavaScript rendering first if available
        final_url = url  # Track the final URL after redirects
        if use_js and (driver or thread_driver):
            try:
                # Create thread-local driver if requested
                if thread_driver:
                    local_driver = setup_selenium_driver(cookie_file)
                    if not local_driver:
                        print(f"Failed to create thread-local driver for {url}, falling back to requests")
                        resp = fetch(session, url)
                        html_bytes = resp.content
                        html_content = html_bytes.decode('utf-8', errors='ignore')
                    else:
                        html_content, final_url = fetch_with_js(local_driver, url, cookie_file)
                        html_bytes = html_content.encode('utf-8')
                else:
                    # Use shared driver (not recommended for concurrent use)
                    html_content, final_url = fetch_with_js(driver, url, cookie_file)
                    html_bytes = html_content.encode('utf-8')
                
                # Update mission name based on final URL after redirects
                if final_url != url:
                    mission_name = final_url.split("/")[-1]
                    print(f"Redirect detected: {url} -> {final_url}")
            except Exception as e:
                print(f"JS rendering failed for {url}, falling back to requests: {e}")
                resp = fetch(session, url)
                html_bytes = resp.content
                html_content = html_bytes.decode('utf-8', errors='ignore')
        else:
            resp = fetch(session, url)
            html_bytes = resp.content
            html_content = html_bytes.decode('utf-8', errors='ignore')
        
        # Check and accept cookies if consent page is detected
        if has_cookie_consent_page(html_content):
            print(f"Cookie consent page detected for {url}, retrying with Selenium...")
            
            # If we haven't tried JS yet, try with Selenium
            if not (use_js and driver):
                print(f"Selenium not available for {url}, marking as consent page")
            else:
                # Force retry with Selenium
                try:
                    retry_driver = local_driver if local_driver else driver
                    if retry_driver:
                        html_content, final_url = fetch_with_js(retry_driver, url, cookie_file, timeout=45)
                        html_bytes = html_content.encode('utf-8')
                        # Update mission name based on final URL after redirects
                        if final_url != url:
                            mission_name = final_url.split("/")[-1]
                            print(f"Redirect detected on retry: {url} -> {final_url}")
                        
                        # Check again after Selenium attempt
                        if has_cookie_consent_page(html_content):
                            print(f"Still showing consent page after Selenium attempt for {url}")
                except Exception as e:
                    print(f"Selenium retry failed for {url}: {e}")
        
        # Save raw HTML
        with open(html_path, 'wb') as f:
            f.write(html_bytes)
        
        # Extract and save text
        text_content = clean_text_from_html(html_content)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        # Extract and save structured content
        structured = extract_structured_content(html_content)
        
        with open(tables_path, 'w', encoding='utf-8') as f:
            json.dump(structured['tables'], f, indent=2, ensure_ascii=False)
        
        with open(images_path, 'w', encoding='utf-8') as f:
            json.dump(structured['images'], f, indent=2, ensure_ascii=False)
        
        # Get title
        title = get_title(html_content)
        
        # Check if this was a successful scrape or consent page
        is_consent_page = has_cookie_consent_page(html_content)
        
        return {
            "url": url,
            "final_url": final_url,
            "original_mission_name": original_mission_name,
            "mission_name": mission_name,
            "local_html": str(html_path),
            "local_text": str(text_path),
            "local_tables": str(tables_path),
            "local_images": str(images_path),
            "title": title,
            "sha256": sha256(html_bytes),
            "bytes": len(html_bytes),
            "num_tables": len(structured['tables']),
            "num_images": len(structured['images']),
            "text_length": len(text_content),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "sitemap",
            "js_rendered": use_js and driver is not None,
            "is_consent_page": is_consent_page,
            "success": not is_consent_page,
            "redirected": final_url != url
        }
        
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None
    finally:
        # Clean up thread-local driver
        if local_driver:
            try:
                local_driver.quit()
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description="Fast eoPortal scraper")
    parser.add_argument("--output-dir", type=Path, default="../data/Fast_Method_eoportal",
                       help="Output directory")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of worker threads")
    parser.add_argument("--use-js", action="store_true", default=True,
                       help="Use JavaScript rendering (requires selenium)")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Request timeout in seconds")
    args = parser.parse_args()
    
    print("fast eoPortal Scraper")
    print("========================")
    
    # Setup
    session = make_session(DEFAULT_UA, args.timeout)
    
    # Cookie file for persistence
    cookie_file = args.output_dir / "cookies.json"
    
    # Pre-accept cookies for the session
    print("Pre-accepting cookies...")
    accept_cookies(session, "https://www.eoportal.org")
    
    # Note: We'll create thread-local drivers instead of a shared one
    if args.use_js:
        # Test if Selenium is available by creating and closing a test driver
        test_driver = setup_selenium_driver(cookie_file)
        if test_driver:
            test_driver.quit()
            print("✓ Selenium WebDriver available - will create thread-local drivers")
            if cookie_file.exists():
                print("✓ Cookie file exists")
        else:
            print("⚠ Selenium not available, using requests only")
            args.use_js = False
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw_html" / "missions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "text" / "missions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "structured" / "missions").mkdir(parents=True, exist_ok=True)
    
    # Discover URLs
    print("\n1. Discovering mission URLs...")
    urls = discover_from_sitemap(session)
    
    if not urls:
        print("No URLs discovered. Exiting.")
        return
    
    print(f"Found {len(urls)} mission URLs")
    
    # Download missions
    print(f"\n2. Downloading missions (max_workers={args.max_workers})...")
    manifest_path = args.output_dir / "manifest_fast.jsonl"
    
    successful = 0
    failed = 0
    consent_pages = 0
    seen_missions = set()  # Track missions we've already downloaded
    
    # Load existing manifest to avoid re-downloading
    if manifest_path.exists():
        print("\nLoading existing manifest to skip already downloaded missions...")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Track both original and final mission names
                    seen_missions.add(entry.get('mission_name', ''))
                    seen_missions.add(entry.get('original_mission_name', ''))
                    if entry.get('final_url') and entry['final_url'] != entry.get('url'):
                        # Also track the final URL's mission name
                        final_mission = entry['final_url'].split('/')[-1]
                        seen_missions.add(final_mission)
                except:
                    continue
        print(f"Found {len(seen_missions)} already downloaded missions")
    
    with open(manifest_path, 'a', encoding='utf-8') as manifest_f:
        # Process URLs in batches to manage memory with Selenium
        batch_size = args.max_workers
        url_list = list(urls)
        
        for i in range(0, len(url_list), batch_size):
            # Filter out URLs for missions we've already downloaded
            batch = []
            for url in url_list[i:i + batch_size]:
                mission_name = url.split('/')[-1]
                if mission_name not in seen_missions:
                    batch.append(url)
                else:
                    print(f"Skipping already downloaded mission: {mission_name}")
            
            if not batch:
                continue
            
            with cf.ThreadPoolExecutor(max_workers=min(args.max_workers, len(batch))) as executor:
                # For concurrent execution, each thread needs its own driver
                future_to_url = {
                    executor.submit(download_mission_fast, session, None, url, args.output_dir, cookie_file, args.use_js, thread_driver=True): url
                    for url in batch
                }
                
                for future in tqdm(cf.as_completed(future_to_url), total=len(batch), desc=f"Batch {i//batch_size + 1}"):
                    result = future.result()
                    if result:
                        # Check if we've already downloaded this mission (in case of redirects)
                        mission_name = result.get('mission_name', '')
                        if mission_name not in seen_missions:
                            manifest_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            manifest_f.flush()
                            # Add to seen missions
                            seen_missions.add(mission_name)
                            seen_missions.add(result.get('original_mission_name', ''))
                            if result.get('is_consent_page', False):
                                consent_pages += 1
                            else:
                                successful += 1
                        else:
                            print(f"Skipping duplicate mission from redirect: {mission_name}")
                    else:
                        failed += 1
    
    # Drivers are now thread-local and cleaned up in download_mission_fast
    
    print(f"\n✓ Scraping complete!")
    print(f"  - Successful: {successful}")
    print(f"  - Cookie consent pages: {consent_pages}")
    print(f"  - Failed: {failed}")
    print(f"  - Output: {args.output_dir}")
    print(f"  - Manifest: {manifest_path}")

if __name__ == "__main__":
    main()

