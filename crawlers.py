import json
import os
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from model import (
    Class,
    Docs,
    DocsLink,
    Enum,
    EnumValue,
    Example,
    Function,
    Link,
    Module,
    Parameter,
)

CRAWL_SINGLE_DOC_LLM_INSTRUCTION = (
    "Extract the functions , modules , classes , example code snippets and there usecases anything which is part of documentation which will help user to understand this documentation"
    "so that user or LLM can understand and write code using them"
    "strictly follow the schema provided"
)

CRAWL_SITEMAP_LLM_INSTRUCTION = (
    "Extract the links of the documentation from the sitemap"
    "strictly follow the schema provided"
)


def save_crawl_data(
    data, prefix: str = "crawl_data", output_dir: str = "output"
) -> str:
    """Save crawled data to file with timestamp"""
    # Create output directory if not exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}"

    try:
        if isinstance(data, dict):
            # Save as JSON
            filepath = os.path.join(output_dir, f"{filename}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            # Save as text
            filepath = os.path.join(output_dir, f"{filename}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(data))
        return filepath
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return None


async def crawl_single_doc(url: str):
    llm_strat = LLMExtractionStrategy(
        provider="gemini/gemini-1.5-flash",
        api_token=os.getenv("GEMINI_API_KEY"),
        schema=[
            Module.schema_json(),
            Class.schema_json(),
            Function.schema_json(),
            Enum.schema_json(),
            EnumValue.schema_json(),
            Link.schema_json(),
            Parameter.schema_json(),
            Example.schema_json(),
            Docs.schema_json(),
        ],
        extraction_type="schema",
        instruction=CRAWL_SINGLE_DOC_LLM_INSTRUCTION,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0},
    )
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strat, cache_mode=CacheMode.BYPASS, scan_full_page=True
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=crawl_config,
        )
        if result.success:
            data = json.loads(result.extracted_content)
            llm_strat.show_usage()
            saved_path = save_crawl_data(data, prefix="single_docs")
            if saved_path:
                print(f"Data saved to: {saved_path}")
        else:
            print(result.error_message)


async def crawl_sitemap(sitemap_url: str):
    llm_strat = LLMExtractionStrategy(
        provider="gemini/gemini-1.5-flash",
        api_token=os.getenv("GEMINI_API_KEY"),
        schema=[
            DocsLink.schema_json(),
        ],
        extraction_type="schema",
        instruction=CRAWL_SITEMAP_LLM_INSTRUCTION,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0},
    )
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strat, cache_mode=CacheMode.BYPASS
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=sitemap_url,
            config=crawl_config,
        )
        if result.success:
            data = json.loads(result.extracted_content)
            llm_strat.show_usage()
            print(data)
            print(len(data))
            saved_path = save_crawl_data(data, prefix="docs_sitemap")
            docs_urls = []
            for item in data:
                print(item["url"])
                docs_urls.append(item["url"])
            if saved_path:
                print(f"Data saved to: {saved_path}")
            for url in docs_urls:
                await crawl_single_doc(url)
        else:
            print(result.error_message)
