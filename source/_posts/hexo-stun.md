---
title: hexo+stun 博客搭建
top_image: 2kjegy.png
toc: true
comments: true
math: true
date: 2021-01-26 23:13:43
tags: 博客搭建
categories: 教程类
---

# 背景

看到某位同学搭了自己的博客，觉得有一个记录自己成长路线的博客好像挺香的…刚好寒假时间比较充裕，是个动手的好时机~

Github Pages 功能 +  Hexo 的博客框架 + 自己喜欢的主题 = 华而有实的博客

<!--more-->

使用 Github Pages 搭建博客的好处：

1. 免费，免费，免费！
2. 都是静态文件，访问速度较优；
3. 能和 git 搭配使用，管理起来都是熟悉的配方；

使用 hexo 的好处：

1. 简洁高效，安装简单；
2. 有多种稳定、美观的主题可以挑选；
3. 使用 Markdown 解析文章，符合自己平时写东西的习惯；

> 当然其他框架也有各自的优点，这个选择凭各自喜好啦~

**本教程以 windows 为例**


# Github Pages

**基于 [Github Pages 官方文档](https://docs.github.com/cn/github/working-with-github-pages/creating-a-github-pages-site)**

前提是要有 github 账号，没有的同学们可以先去注册一个~

1. 创建 Github Pages 站点：

    - 新建一个仓库；
    - 输入仓库的名称和说明（可选）。 如果您创建的是用户或组织站点，仓库名称**必须为** `<user>.github.io` 或 `<organization>.github.io` ；
    - 设置仓库属性为 `public` ;
    - 选择 `Initialize this repository with a README`（使用 readme 文件初始化此仓库；
    - `create repository`

2. 设置站点：

    - 在站点仓库下，找到右上角 `Settings`

    - 下滑，找到`Github Pages`选项卡，你能看到站点的 url 啦~
        {% asset_img 1-1.jpg This is an github-pages image %}
        > 这个 url 就是你网站的地址，通过它，大家就能访问到你的博客。

    - Source 栏的 branch 选择 main（或者 master，总之就是除了 none 之外的那个），点击 save；

    - Theme Chooser 那个不用管，因为后续要使用 hexo 的主题嘛；

    - 将仓库克隆到本地，用 git 配置自己的身份信息：

        ```
        git config --global user.name "github user name"
        git config --global user.email "github user email"
        ```

    - [添加 ssh-key](https://www.liaoxuefeng.com/wiki/896043488029600/896954117292416)（注意不要设置密码）主要是为了以后 hexo 发布比较方便。



# hexo

**基于 [hexo](https://hexo.io/zh-cn/docs/) 官方文档**

## 安装

安装 hexo 之前，先来把环境搞好：

- 安装 [Node.js](http://nodejs.org/) (Node.js 版本需不低于 10.13，建议使用 Node.js 12.0 及以上版本)
- 安装 Git （ 不过 github 那一步都搞好了，这里就不用了吧~）

> 这里具体的安装细节提示可以直接参考各自官方文档，或者网上找找教程。

现在可以安装 hexo。

在合适的地方新建一个文件夹，用来存放自己的博客文件，比如我的博客文件都存放在`D:/blog`目录下。

我使用 Windows 自带的控制台定位到`D:/blog`目录下执行以下操作：

```
$npm install -g hexo-cli
```

> 据说 linux 和 max 里是要在前面加个 sudo，不然会因为权限问题报错。

装完输入 `hexo --version` 检查是否安装成功。

## 建站

`hexo init` 初始化文件夹；

`npm install` 安装必须的插件；

`hexo g` 生成静态文件；

`hexo s` 将静态文件运行在本地服务器上，这个时候根据提示打开 `localhost:4000` 就能看到最基本的博客啦~

`ctrl+c` 关闭本地服务器；

## 与 github 连接

打开 blog 根目录下的 `_config.yml` 文件，修改配置：

```yml
# URL
## If your site is put in a subdirectory, set url as 'http://example.com/child' and root as '/child/'
url: https://github.com/xxx/xxx.github.io/
root: /xxx.github.io/
permalink: :year/:month/:day/:title/
permalink_defaults:
pretty_urls:
  trailing_index: true # Set to false to remove trailing 'index.html' from permalinks
  trailing_html: true # Set to false to remove trailing '.html' from

...

# Deployment
## Docs: https://hexo.io/docs/one-command-deployment
deploy:
  type: git
  repo: git@github.com:xxx/xxx.github.io.git
  branch: main
```

还是在 blog 根目录下：

```c
hexo g // 编译生成静态文件（每次修改完都必须重新编译）
hexo d // 将博客发布到 github 上
```

现在可以通过 `https://xxx.github.io` 来访问你的博客啦~



# 主题 - stun

博主选的是 stun 这个主题，主要是看中了她 ~~可甜可盐~~  活泼大方、简洁美观 的风格，当然她不是最简洁的，最简洁的应该是 Next 这个主题了吧（

这个官方文档超级全的！！！而且步骤都很详细！！！

我在这里就不瞎写了，反正写的没有官网好，照着来没错的，嗯！

官网指路：[stun](https://theme-stun.github.io/docs/zh-CN/guide/quick-start.html#%E5%AE%89%E8%A3%85)


emmmm

那我写点推荐和排雷：

## 统计与分析推荐

首推 [谷歌分析](https://theme-stun.github.io/docs/zh-CN/advanced/third-part.html#%E8%B0%B7%E6%AD%8C%E5%88%86%E6%9E%90) ！！！ 简单好用不要钱！！！

来一篇好用的配置教程：[Google Analytics怎么用，谷歌分析工具使用教程](https://www.yundianseo.com/how-to-use-google-analytics/)



## 评论系统推荐

博主尝试了三种评论系统，各自利弊写在下面奥：

1. Disqus

    优点：

    - 配置简单（所以博主最先选的就是这个）
    - 完善的后台管理机制
    - 丰富的表情可选
    - 支持 markdown

    缺点：

    - 服务器在国外，不翻墙加载不出来

    - 存在广告植入

    - 要评论必须有 disqus  / google / Twitter/ facebook 账户



2. Valine

    优点：

    - 配置简单（但没有 disqus 简单）
    - 无后端，所以加载起来很快
    - 页面设计简洁
    - 评论不用登陆任何账户
    - 支持 markdown

    缺点：

    - 页面设计过于简洁（我试验过之后才发现，是真的很简洁，白白的，也没有点赞的功能，想要拥有一个头像都要花挺多功夫
    - 评论可以匿名（某同学说可能会有恶意评论，博主觉得他说的有道理…



3. Utterances

    优点：

    - 配置看上去复杂但其实很简单：

        将 [utterances app (opens new window)](https://github.com/apps/utterances)安装在你博客对应的 Github 仓库中。然后，按照 stun 官网的提示修改配置项即可；

    - 是一个基于 github issues 的评论系统，管理方便；

    - 支持 emoji 支持评论点赞

    - 支持 markdown

    缺点：

    - 需要登录 github 账户才能评论（是缺点也是优点吧）

## bug 分享

### multiline key

```
 message: 'can not read a block mapping entry; a multiline key may not be an implicit key at line 7, column 9:\n'
 "    subtitle: 'If you shed tears when you mi ... \n" +
```

网上大部分说是在对应位置缺了英文空格，但我不是这个错…

这里要注意，错误可能不在这一行，可能出现在它前一行：

```yml
title: 'Karin Lv's Blog'
subtitle: 'If you shed tears...
```

我就是出现在它上一行的，这里单引号中间还有一个单引号，不符合语法规则了，所以报错…

去掉外层的单引号就好了！

### invalid characters

出现非法字符，大概率是因为在配置文件里写了中文，但没有相应的设置。

博主的做法比较简单粗暴：配置文件里尽量不写中文，都改成英文 QAQ

> 大家可以在网上搜一下更专业的解决办法，不要学这个懒博主（

