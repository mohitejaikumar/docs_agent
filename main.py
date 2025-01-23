import asyncio
import json
import os
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from pathlib import Path
from datetime import datetime
from crawl4ai.extraction_strategy import LLMExtractionStrategy


class Link(BaseModel):
    id: int
    source_type: Literal["function", "class", "module", "enum", "docs", "library"]
    source_id: int
    target_type: Literal["function", "class", "module", "enum", "docs", "library"]
    target_id: int
    type: Optional[str] = None


class Parameter(BaseModel):
    name: str = Field(
        ..., description="Parameter name, can be for class constructor or function"
    )
    type: Optional[str] = None
    docs_id: int
    description: Optional[str] = None
    default_value: Optional[str] = None
    is_optional: Optional[bool] = None


class Example(BaseModel):
    type: str = Field(
        ...,
        description="type is to describe the usecase of that code snippet , task which we can accomplish using this code",
    )
    code: str = Field(..., description="Example code snippet")
    description: Optional[str] = None


class Function(BaseModel):
    id: int
    module_id: Optional[int] = None
    docs_id: int
    class_id: Optional[int] = None
    name: str = Field(
        ...,
        description="Function/method name it can be associated with class or it can be pure function from module",
    )
    description: Optional[str] = None
    signature: Optional[str] = None
    parameters: Optional[List[Parameter]] = None
    examples: Optional[List[Example]] = None
    url: Optional[str] = None


class Class(BaseModel):
    id: int
    module_id: int
    docs_id: int
    name: str = Field(..., description="Class we that we export from a module")
    description: Optional[str] = None
    url: Optional[str] = None


class Enum(BaseModel):
    id: int
    module_id: int
    docs_id: int
    name: str = Field(..., description="Enum name")
    description: Optional[str] = None
    url: Optional[str] = None


class EnumValue(BaseModel):
    id: int
    enum_id: int
    docs_id: int
    name: str = Field(..., description="Enum value name")
    value: Optional[str] = None
    description: Optional[str] = None


class Module(BaseModel):
    id: int
    library_id: int
    docs_id: int
    name: str = Field(..., description="Name of the module")
    description: Optional[str] = None
    url: Optional[str] = None


class Library(BaseModel):
    id: int
    name: str = Field(..., description="Name of the library")
    version: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None


class Docs(BaseModel):
    id: int
    library_id: int
    url: Optional[str] = None
    description: str = Field(
        ...,
        description="Includes the additional information helps to clearly understand this doc which we are scrapping",
    )


example_data = {
    "library": [
        {
            "id": 1,
            "name": "Crawl4AI",
            "version": "v0.4.3b2",
            "url": "https://docs.crawl4ai.com/advanced/file-downloading/",
            "description": "Crawl4AI is a library for handling file downloads during web crawling.",
        }
    ],
    "docs": [
        {
            "id": 1,
            "library_id": 1,
            "url": "https://docs.crawl4ai.com/advanced/file-downloading/",
            "description": "This guide explains how to use Crawl4AI to handle file downloads during crawling. You'll learn how to trigger downloads, specify download locations, and access downloaded files.Important Considerations Browser Context: Downloads are managed within the browser context. Ensure js_code correctly targets the download triggers on the webpage.Timing: Use wait_for in CrawlerRunConfig to manage download timing.Error Handling: Handle errors to manage failed downloads or incorrect paths gracefully.Security: Scan downloaded files for potential security threats before use.This revised guide ensures consistency with the Crawl4AI codebase by using BrowserConfig and CrawlerRunConfig for all download-related configurations. Let me know if further adjustments are needed!",
        }
    ],
    "modules": [
        {
            "id": 1,
            "library_id": 1,
            "docs_id": 1,
            "name": "crawl4ai.async_configs",
            "description": "Module for configuring asynchronous web crawling.",
            "url": "https://docs.crawl4ai.com/advanced/file-downloading/",
        }
    ],
    "classes": [
        {
            "id": 1,
            "module_id": 1,
            "docs_id": 1,
            "name": "BrowserConfig",
            "description": "Class for configuring the browser for web crawling.",
            "url": "https://docs.crawl4ai.com/advanced/file-downloading/",
        },
        {
            "id": 1,
            "module_id": 1,
            "docs_id": 1,
            "class_id": 1,
            "name": "AsyncWebCrawler",
            "description": "Class for asynchronous web crawling.",
            "url": "https://docs.crawl4ai.com/advanced/file-downloading/",
            "signature": "AsyncWebCrawler(config)",
            "parameters": [
                {
                    "name": "config",
                    "type": "BrowserConfig",
                    "docs_id": 1,
                    "description": "Configuration for the browser.",
                    "default_value": "null",
                }
            ],
            "examples": [
                {
                    "type": "Enabling Downloads",
                    "code": "from crawl4ai.async_configs import BrowserConfig, AsyncWebCrawler\\nasync def main():\\n  config = BrowserConfig(accept_downloads=True) # Enable downloads globally\\n  async with AsyncWebCrawler(config=config) as crawler:\\n    # ... your crawling logic ...\\nasyncio.run(main())",
                    "description": "This example shows how to enable downloads globally using the `BrowserConfig` and `AsyncWebCrawler` classes.",
                },
                {
                    "type": "Specifying Download Location",
                    "code": """from crawl4ai.async_configs import BrowserConfig
                    import os
                    downloads_path = os.path.join(os.getcwd(), "my_downloads") # Custom download path
                    os.makedirs(downloads_path, exist_ok=True)
                    config = BrowserConfig(accept_downloads=True, downloads_path=downloads_path)
                    async def main():
                    async with AsyncWebCrawler(config=config) as crawler:
                        result = await crawler.arun(url="https://example.com")
                        # ...""",
                    "description": "This example shows how to specify a custom download location using the `BrowserConfig` class.",
                },
            ],
        },
    ],
    "functions": [
        {
            "id": 1,
            "module_id": 1,
            "docs_id": 1,
            "class_id": 1,
            "name": "arun",
            "description": "Method for running the asynchronous web crawler",
            "signature": "arun(url, config)",
            "parameters": [],
            "examples": [
                {
                    "type": "Triggering Downloads",
                    "code": """from crawl4ai.async_configs import CrawlerRunConfig\\nconfig = CrawlerRunConfig(\\n  js_code=\\\\"\\n    const downloadLink = document.querySelector('a[href$=\\\".exe\\\"]\');\\n    if (downloadLink) {\\n      downloadLink.click();\\n    }\\n  \\\",\\n  wait_for=5 # Wait 5 seconds for the download to start\\n)\\nresult = await crawler.arun(url=\\"https://www.python.org/downloads/\\", config=config)""",
                    "description": "This example shows how to trigger downloads by simulating user interactions on a web page using the `CrawlerRunConfig` class.",
                },
                {
                    "type": "Accessing Downloaded Files",
                    "code": """if result.downloaded_files:\\n  print(\\"Downloaded files:\\")\\n  for file_path in result.downloaded_files:\\n    print(f\\"- {file_path}\\")\\n    file_size = os.path.getsize(file_path)\\n    print(f\\"- File size: {file_size} bytes\\")\\nelse:\\n  print(\\"No files downloaded.\\")""",
                    "description": "This example shows how to access downloaded files using the `downloaded_files` attribute of the `CrawlResult` object.",
                },
                {
                    "type": "Downloading Multiple Files",
                    "code": """from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig\\nimport os\\nfrom pathlib import Path\\nasync def download_multiple_files(url: str, download_path: str):\\n  config = BrowserConfig(accept_downloads=True, downloads_path=download_path)\\n  async with AsyncWebCrawler(config=config) as crawler:\\n    run_config = CrawlerRunConfig(\\n      js_code=\\\\"\\n        const downloadLinks = document.querySelectorAll('a[download]');\\n        for (const link of downloadLinks) {\\n          link.click();\\n          // Delay between clicks\\n          await new Promise(r => setTimeout(r, 2000)); \\n        }\\n      \\\",\\n      wait_for=10 # Wait for all downloads to start\\n    )\\n    result = await crawler.arun(url=url, config=run_config)\\n    if result.downloaded_files:\\n      print(\\"Downloaded files:\\")\\n      for file in result.downloaded_files:\\n        print(f\\"- {file}\\")\\n    else:\\n      print(\\"No files downloaded.\\")\\n# Usage\\ndownload_path = os.path.join(Path.home(), \\".crawl4ai\\", \\"downloads\\")\\nos.makedirs(download_path, exist_ok=True)\\nasyncio.run(download_multiple_files(\\"https://www.python.org/downloads/windows/\\", download_path))""",
                    "description": "This example shows how to download multiple files using the `BrowserConfig` and `CrawlerRunConfig` classes.",
                },
            ],
            "url": "https://docs.crawl4ai.com/advanced/file-downloading/",
        }
    ],
}

example_string = json.dumps(example_data, indent=2)

LLM_INSTRUCTION = (
    "Extract the functions , modules , classes , example code snippets and there usecases anything which is part of documentation which will help user to understand this documentation"
    "so that user or LLM can understand and write code using them"
    "strictly follow the schema provided"
    "output_example: {example_string}"
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


async def main():
    llm_strat = LLMExtractionStrategy(
        provider="openai/gpt-4",
        api_token=os.getenv("OPENAI_API_KEY"),
        schema=[
            Library.schema_json(),
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
        instruction=LLM_INSTRUCTION,
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
            url="https://docs.crawl4ai.com/advanced/hooks-auth/",
            config=crawl_config,
        )
        if result.success:
            data = json.loads(result.extracted_content)
            llm_strat.show_usage()
            saved_path = save_crawl_data(data, prefix="crawl4ai_docs")
            if saved_path:
                print(f"Data saved to: {saved_path}")
        else:
            print(result.error_message)


if __name__ == "__main__":
    asyncio.run(main())
