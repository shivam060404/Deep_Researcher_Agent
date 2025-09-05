from langchain_tavily import TavilySearch

web_search_tool = TavilySearch(
    description="A general-purpose web search engine for finding information from websites, blogs, and forums."
)

academic_search_tool = TavilySearch(
    description="A search engine optimized for academic papers, journals, and scientific articles."
)

news_search_tool = TavilySearch(
    description="A search engine specialized for news articles, current events, and media coverage."
)

tools = [web_search_tool, academic_search_tool, news_search_tool]
