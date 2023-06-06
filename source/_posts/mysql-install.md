---
title: MySQL 安装教程 + 排坑
toc: true
comments: true
math: true
date: 2021-01-27 22:58:11
tags: mysql
categories: 教程类
---
**本教程基于 windows10(64位) 操作系统**
<!--more-->
# 下载

1. 访问 [官网下载地址](https://dev.mysql.com/downloads/mysql/)
   {% asset_img 1.PNG This is an download image %}

2. 点击 `No thanks,just start my download` 跳过注册过程，直接下载压缩包。


# 安装与配置

- 在指定路径解压，将解压后 bin 文件的路径（`D:\mysql-8.0.23-winx64\bin`）添加到系统环境变量中：
  > 我的电脑->属性->高级->环境变量
  > 选择PATH,在其后面添加: 你的 mysql bin 文件夹的路径
- 使用管理员身份打开 cmd：
  > 在开始菜单中输入 cmd，选择用管理员身份打开；
- 跳转到 mysql bin 目录下：
  {% asset_img 2.PNG This is an bin path %}
- 安装 mysql
  ```
  mysqld -install
  ```
- 初始化 mysql（一定要初始化！否则容易导致启动不成功）
  ```
  mysqld --initialize
  ```
- 启动服务
  ```
  net start mysql
  ```
  {% asset_img 3.PNG This is an start successfully image %}

  > 这里直接写 mysql 是因为我主机服务中 mysql 的服务名就是 mysql，如果大家出现了启动不成功的情况可以自行搜索是不是服务名不同。

- 登录 mysql：
  ```
  mysql -u root -p
  ```
  据说第一次登录不需要密码，直接按回车就好了，但是博主按了回车发现不行。出现报错 `ERROR 1045 (28000): Access denied for user 'root'@'localhost' (using password: NO)`
  {% asset_img 4.PNG This is an start failed image %}
  看来还是需要我们输入密码，下面一起找密码吧：
  1. 打开 mysql 的根目录下名为 data 的文件夹；
  2. 找到以 `.err` 为结尾的文件（专门记录报错信息的）并打开；
  3. 找到自动生成的密码啦！
   {% asset_img 5.PNG This is an password image %}
  4. 输入密码即可~
   {% asset_img 6.PNG This is an load successfully image %}

# 通过 vscode 连接 mysql

旨在利用 vscode 能更加便捷的编写并执行 SQL 代码。

参考文章：[MySQL vscode开发环境搭建](https://zhuanlan.zhihu.com/p/347159257)

## 基础知识

> MySQL 相当于一个 shell，SQL 就是和 shell 交互的脚本语言。
> 众所周知，与 shell 打交道的方式有两种：
> - 一种是直接在shell中输命令执行，但是这样很难让我们看到多条shell的作用。
> - 还有一种方式是编写 shell 脚本。在 bash 中，shell 脚本的文件名后缀可以是 .sh，在 MySQL 中，脚本的后缀名为 .sql

## 安装插件

1. 打开 vscode 插件商店；
2. 搜索并安装如下两个插件：
   - MySQL [by Jun Han]
   - MySQL Syntax [by Jake Bathman]
3. 以管理员身份打开 cmd；
4. `net start mysql` 启动服务；
5. `mysql -u root -p` 进入 MySQL 账户（密码的查找方式见上）
6. 进入 mysql 的 shell 之后，输入命令：
   ```
    alter user 'root'@'localhost' identified with mysql_native_password by '123456';
   ```
   重置密码为 `123456`

## vscode 连接本机数据库

1. 打开 vscode ，点击左下角的 MYSQL 旁边的加号：
   {% asset_img 7.PNG This is an load successfully image %}

2. 在弹出的对话框中设置参数：
   - host: 127.0.0.1
   - username: root
   - password: 123456 (刚设置的)
   - 其他参数保持默认，一路回车下去

## 一个栗子

1. 在 MYSQL 下方的蓝色饼饼图标处右键，选择 new_query
2. 在弹出的编辑栏中输入 SQL 指令，并以 `.sql` 为后缀名保存到本地
3. 编辑区右键，选择 Run MySQL query
4. 你会在编辑区右侧看到指令的执行结果。

## 震惊！vscode 的 mysql 插件不支持过程化 sql 语句

太悲惨了，博主爆哭 wwwww
为了不放弃自己辛辛苦苦搞的 vscode ，博主进行了多方尝试，最终还是不行 wwwwwww
大家转战 mysql workbench 吧（官网下载安装）


