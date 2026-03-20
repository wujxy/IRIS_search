计划目标
为现有的文献数据库系统补充两个核心模块：

文献展示模块：将SQLite数据库中的文献通过网页形式展示，提供文献列表和详情查看功能

接口模块：为已实现的模型问答功能提供标准化的API接口，供前端调用

技术选型
Web框架：FastAPI

模板引擎：Jinja2

前端样式：原生CSS（简洁学术风格）

API文档：FastAPI自动生成

开发任务清单
任务一：创建基础应用结构
创建FastAPI主应用文件

配置模板和静态文件目录

设置数据库连接（复用现有数据库）

任务二：文献展示模块
文献列表页

分页显示文献（每页10条）

展示标题、作者、年份、期刊、摘要预览

卡片式布局，响应式设计

文献详情页

显示完整文献信息（标题、作者、摘要、期刊、年份、DOI、由生成模型给出的完整summary等）

提供返回列表按钮

基础模板（base.html）

统一的头部导航栏和底部信息

包含搜索框（调用现有检索功能）

推荐文件目录结构：（仅供参考，结合实际plan修改）
IRIS_search/
├── src/
│   ├── infrastructure/          # 基础设施层（已有）
│   ├── core/                   # 核心业务逻辑层（已有）
│   ├── services/               # 业务服务层（已有）
│   └── web/                    # 新增：Web展示模块
│       ├── __init__.py
│       ├── app.py              # FastAPI主应用
│       ├── routers/            # 路由模块
│       │   ├── __init__.py
│       │   ├── web_routes.py   # 网页展示路由
│       │   └── api_routes.py   # API接口路由
│       ├── models.py           # Pydantic数据模型
│       ├── templates/          # Jinja2模板
│       │   ├── base.html       # 基础模板
│       │   ├── index.html      # 文献列表页
│       │   ├── detail.html     # 文献详情页
│       │   └── error.html      # 错误页面
│       └── static/             # 静态资源
│           ├── css/
│           │   └── style.css
│           └── js/
│               └── main.js
├── scripts/
│   ├── run_update_cycle.py     # 主编排脚本（已有）
│   ├── iris_query.py          # 查询接口（已有）
│   └── run_web.py             # 新增：启动Web服务脚本
└── utils/                      # 工具函数（已有）

未来可能需要在网页增加与生成模型互动问答的模块（python scripts/iris_query.py --interactive的功能），请预留接口 [本次任务需要增加模型互动模块，仅集中于文献展示]