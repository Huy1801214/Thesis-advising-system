import os
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

# Cố gắng import BeautifulSoup, nếu không có sẽ tự viết hàm clean_html đơn giản
try:
    from bs4 import BeautifulSoup
    def clean_html(html_content: str) -> str:
        return BeautifulSoup(html_content, "html.parser").get_text()
except ImportError:
    import re
    def clean_html(html_content: str) -> str:
        # Xóa các tag html cơ bản
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html_content)

class TechCrawler:
    def __init__(self):
        # Các nguồn RSS feed thực tế cực kỳ uy tín và ổn định
        self.feeds = [
            "https://aws.amazon.com/blogs/aws/feed/",
            "https://devblogs.microsoft.com/python/feed/",
            "https://github.blog/feed/"
        ]
        self.output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "crawled_articles"

    def crawl_feeds(self, limit_per_feed: int = 3) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        crawled_count = 0
        
        # Xóa các file cũ trong thư mục cào trước khi chạy
        for f in self.output_dir.glob("*.txt"):
            try:
                f.unlink()
            except Exception:
                pass

        for feed_url in self.feeds:
            try:
                print(f"[Crawler] Đang cào dữ liệu từ RSS Feed: {feed_url}")
                # Giả danh User-Agent để tránh bị một số Cloudflare chặn
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(feed_url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    print(f"[Crawler] Lỗi HTTP {resp.status_code} khi tải {feed_url}")
                    continue
                
                # Parse cấu trúc XML của RSS
                root = ET.fromstring(resp.content)
                items = root.findall(".//item")
                
                for idx, item in enumerate(items[:limit_per_feed]):
                    title_el = item.find("title")
                    link_el = item.find("link")
                    desc_el = item.find("description")
                    pub_el = item.find("pubDate")
                    
                    title = title_el.text if title_el is not None else "Không có tiêu đề"
                    link = link_el.text if link_el is not None else ""
                    pub_date = pub_el.text if pub_el is not None else ""
                    description_raw = desc_el.text if desc_el is not None else ""
                    
                    # Làm sạch nội dung HTML
                    content_text = clean_html(description_raw)
                    
                    # Tạo file lưu trữ cục bộ
                    filename = f"article_{crawled_count}.txt"
                    filepath = self.output_dir / filename
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"Source: {feed_url}\n")
                        f.write(f"Title: {title}\n")
                        f.write(f"Link: {link}\n")
                        f.write(f"Date: {pub_date}\n")
                        f.write(f"Content:\n{content_text.strip()}\n")
                    
                    print(f"   -> Đã cào thành công: {title[:50]}...")
                    crawled_count += 1
            except Exception as e:
                print(f"[Crawler] Lỗi cào nguồn {feed_url}: {e}")
        
        return crawled_count

if __name__ == "__main__":
    crawler = TechCrawler()
    crawler.crawl_feeds()
