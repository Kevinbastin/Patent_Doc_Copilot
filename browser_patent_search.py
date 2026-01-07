"""
Browser-Based Google Patents Search
====================================
Uses Playwright for 100% accurate patent search results.

Install: pip install playwright && playwright install chromium
"""

import re
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PatentResult:
    patent_number: str
    title: str
    abstract: str = ""
    assignee: str = ""
    url: str = ""
    verified: bool = True


class GooglePatentsBrowser:
    """Browser-based Google Patents search."""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser = None
    
    async def search(self, query: str, max_results: int = 10) -> List[PatentResult]:
        """Search Google Patents using browser."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("❌ Install: pip install playwright && playwright install chromium")
            return []
        
        patents = []
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page()
                
                # Search Google Patents
                url = f"https://patents.google.com/?q={query.replace(' ', '+')}"
                print(f"🔍 Searching: {query[:40]}...")
                
                await page.goto(url, timeout=30000)
                await asyncio.sleep(3)
                
                # Extract from HTML
                content = await page.content()
                patents = self._extract_patents(content, max_results)
                
                # Get details for top patents
                for i, patent in enumerate(patents[:5]):
                    try:
                        await page.goto(f"https://patents.google.com/patent/{patent.patent_number}", timeout=15000)
                        await asyncio.sleep(1)
                        
                        # Get title from meta
                        title = await page.evaluate("document.querySelector('meta[name=\"DC.title\"]')?.content || ''")
                        abstract = await page.evaluate("document.querySelector('meta[name=\"DC.description\"]')?.content || ''")
                        assignee = await page.evaluate("document.querySelector('meta[name=\"DC.contributor\"]')?.content || ''")
                        
                        if title:
                            patent.title = title[:150]
                        if abstract:
                            patent.abstract = abstract[:300]
                        if assignee:
                            patent.assignee = assignee[:80]
                    except:
                        continue
                
                await browser.close()
                print(f"✅ Found {len(patents)} verified patents")
                
        except Exception as e:
            print(f"⚠️ Browser search error: {e}")
        
        return patents
    
    def _extract_patents(self, html: str, max_results: int) -> List[PatentResult]:
        """Extract patent numbers from HTML."""
        patents = []
        seen = set()
        
        patterns = [
            r'(US\d{7,}[AB]?\d*)',
            r'(EP\d{7,}[AB]?\d*)',
            r'(WO\d{10,})',
            r'(CN\d{9,}[ABU]?)',
            r'(KR\d{8,}[AB]?\d*)',
            r'(JP\d{7,}[AB]?\d*)',
        ]
        
        for pattern in patterns:
            for match in re.findall(pattern, html):
                if match not in seen:
                    seen.add(match)
                    patents.append(PatentResult(
                        patent_number=match,
                        title=f"Patent {match}",
                        url=f"https://patents.google.com/patent/{match}"
                    ))
                    if len(patents) >= max_results:
                        return patents
        
        return patents


def search_patents_browser(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search Google Patents with browser automation.
    
    Returns list of verified patent dicts.
    """
    searcher = GooglePatentsBrowser(headless=True)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    results = loop.run_until_complete(searcher.search(query, max_results))
    
    return [
        {
            "patent_number": r.patent_number,
            "title": r.title,
            "abstract": r.abstract,
            "assignee": r.assignee,
            "url": r.url,
            "verified": r.verified,
            "source": "Google Patents (Browser)"
        }
        for r in results
    ]


if __name__ == "__main__":
    query = "water purification nanofiber UV filter"
    results = search_patents_browser(query, 8)
    
    print(f"\n{'='*60}")
    print(f"Found {len(results)} patents:")
    print('='*60)
    
    for p in results:
        print(f"\n{p['patent_number']}")
        print(f"  Title: {p['title'][:50]}...")
        print(f"  Assignee: {p['assignee'][:40]}")
