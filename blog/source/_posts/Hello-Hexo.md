---
title: Hello_Hexo
date: 2020-05-05 22:44:59
tags: Githubpages,Hexo
---

## Hexo+GitHub Pages

第一次使用GitHubpages发布博客，之前是wordpress的动态博客，但由于博客性质适合静态网页，于是跳到两年前埋下伏笔处Githubpages。既不用续域名做备案，也不用租用服务器（虽然牛客网那白嫖了一年华为云的）。

#### jekyll or hexo

​	目前两大静态博客框架：[Jekyll](jekyllcn.com)和[hexo](heox.io)。

- 前者本身是被GitHub采用渲染的，而后者是先生成静态文件再上传，由此分离开来。
- 另一个大区别是依赖，前者需要本地安装ruby等而后者需要nodejs，他们的环境配置都有相应文档，另外hexo中文文档对我这种低水平英语很贴心

#### 部署

- 经过探索参考博客，纠正了自己几个错误认知：
  1. hexo初始化后文件夹位置自由，所以如果把他放到一个repo就能通过clone来多端写文（我是放在user-page的repo），这样既能管理blog也能做其他静态web项目（如HX）。
  2. 我之前有个错误认知是只能访问user-page（也就是username.github.io），其实每个repo都能设置page（如username.github.io/blog，它是blog库的一个分支），它需要的是一个分支。而userpage只能是master分支。。
  3. 需要deploy时需npm下载git的包，站点配置文件的deploy项改为对应的repo和对应分支（国内gitee同样）。
- 本站使用的Next主题。先据hexo文档配置好hexo环境，再下载主题到配置文件theme中，通过git clone _repo_url_  theme/next 一步完成也可下载压缩包解压到theme目录下。修改站点配置文件的theme字段即可。需要查看next的文档来配置，继而有下面的特效美化 

## 博客美化（待续）

#### 评论系统

- 基于github-issue的[gitment](<https://imsun.net/posts/gitment-introduction/>)

- 此外有韩国[来必力](livere.com)。在next主题中使用参见[文档](<http://theme-next.iissnan.com/third-party-services.html>)第三方服务。

#### 数据统计与分析-在上面文档

#### 看板娘

- [live2d](<https://www.cnblogs.com/ButterflyEffect/p/10839613.html>)

#### 点击特效

- 光标 点击烟花[这里](<https://www.cnblogs.com/axqa/p/11537599.html>)
- 以上在百度知乎GitHub上找

#### 更多特效

- [播放器头像旋转等](<https://blog.csdn.net/u011475210/article/details/79023429#comments>)、[next深度定制](<https://blog.csdn.net/weixin_43738731/article/details/85843474>)

#### 解决时延问题

![需要根据public结构](https://kivid.github.io/blog/image/解决时延.png)

- 发布后发现加载过慢于是看看是什么原因。第一个是font即谷歌字体的镜像网站无法访问，本来查了些镜像站发现不会用，而原本默认的居然能访问于是在主题配置文件中更改font字段下的host。
- 第二个是disq第三方评论网站被墙，只需将其字段设false。
- 还有个是head的背景图片url失效去主题博主网站下载了（不知道为什么能Get到可能是本地的）。但本地应用由于hexo生成public文件夹发布会找不到于是使用其url（在head-style文件）
- Aplayer是js没有找到，查看博客的repo文件。
- 网易云外链在博客只能点击几首歌，在这里发现是外链问题。依次对单曲产生外链发现是版权问题。