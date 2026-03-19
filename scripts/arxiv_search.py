import arxiv

# 定义查询关键词列表
query_list = [
    "neutrino",
    "juno",
    "dune",
    "hyper-kamiokande",
    "super-kamiokande",
    "icecube",
    "sno",
    "kamland",
    "borexino",
]

# 使用 OR 连接所有条件
query = " OR ".join([f"\"{item}\"" if ' ' in item else item for item in query_list])

search = arxiv.Search(
  query = query,
  max_results = 10,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

print(f"搜索查询: {query}\n")
# check entry_id, updated, published, title, authors, summary, comment, journal_ref, doi, primary_category, categories, pdf_url
for result in search.results():
  print(result.entry_id, '->', result.updated, '->', result.published, '->', result.title, '->', result.authors, '->', result.summary, '->', result.comment, '->', result.journal_ref, '->', result.doi, '->', result.primary_category, '->', result.categories, '->', result.pdf_url)
  
# 下载PDF文件
for result in search.results():
    try:
        # 正确的参数名是 filename（不是 filename_prefix）
        filename = result.entry_id.split('/')[-1] + ".pdf"
        result.download_pdf(dirpath="./test_pdfs", filename=filename)
        print(f"已下载: {result.title}")
    except Exception as e:
        print(f"下载失败: {result.title}, 错误: {e}")