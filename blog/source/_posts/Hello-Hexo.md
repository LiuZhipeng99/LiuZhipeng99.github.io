---
title: Hello_Hexo
date: 2020-05-05 22:44:59
tags: [Githubpages,Hexo]
categories: blog
---

本篇主要介绍如何用Hexo搭建Github Pages，并且对比了同类静态博客框架，记录了部署方面的易错认知，提出了一些常用插件及美化博客的思路。PS：本文非小白教程类
<!--more-->

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

> 下面在next配置文件有接口不用多写代码，看其文档使用方法再进主题配置文件修改，下面主要针对其他类

#### 评论系统

- 此外有韩国[来必力](livere.com)。在next主题中使用参见[文档](http://theme-next.iissnan.com/third-party-services.html)第三方服务来配置。
- 基于github-issue的[gitment](<https://imsun.net/posts/gitment-introduction/>)
- 另找到个valine有使用[文档](https://valine.js.org) 注册用的leancloud可以再创建个app给下面阅读量用
  - 在2020.4.22的更新中：修复使用了低版本的`av-min.js`造成的初始化错误，这也是当前next主题还未修改的。只需要将_src="//cdn1.lncld.net/static/js/3.0.4/av-min.js引入的这个leancloud操作库注释，否则会出现AV对象没有初始化的报错
- PS：摘要使用<!--more-->分割；阅读全文修改主题配置auto_excerpt字段

#### 恶搞标题

- 点击[效果](diygod.me)

- 在主题文件下找到head文件里加入

  ```javascript
   var OriginTitle = document.title;
   var titleTime;
   document.addEventListener('visibilitychange', function () {
       if (document.hidden) {
           $('[rel="icon"]').attr('href', "/img/trhx2.png");
           document.title = 'ヽ(●-`Д´-)ノ人呢！';
           clearTimeout(titleTime);
       }
       else {
           $('[rel="icon"]').attr('href', "/img/trhx2.png");
           document.title = 'ヾ(Ő∀Ő3)ノ回来了！' + OriginTitle;
           titleTime = setTimeout(function () {
               document.title = OriginTitle;
           }, 2000);
       }
   });
  ```

  

#### 看板娘

- [live2d](<https://www.cnblogs.com/ButterflyEffect/p/10839613.html>)

#### 点击滑动特效

- 点击🎈爆炸:放在主题文件下_layout文件中

  ```html
  <!-- 点击出现彩色气球爆炸效果 -->
  <canvas class="fireworks" style="position: fixed;left: 0;top: 0;z-index: 1; pointer-events: none;" ></canvas> 
  <script type="text/javascript" src="//cdn.bootcss.com/animejs/2.2.0/anime.min.js"></script> 
  <script src="https://cdn.jsdelivr.net/gh/wallleap/cdn/js/clickBom.js"></script>
  ```

  

- 光标 点击烟花效果[这里](<https://www.cnblogs.com/axqa/p/11537599.html>)

- 鼠标滑过特效 星星残影：

  ```html
  <script src="https://cdn.jsdelivr.net/gh/wallleap/cdn/js/xuehua.js"></script>
  ```

  

#### 网站运行时

- 在主题footer加入：

  ```html
  <span id="timeDate">载入天数...</span><span id="times">载入时分秒...</span>
  <script>
    var now = new Date(); 
    function createtime() { 
      var grt= new Date("03/08/2020 16:44:00");//此处修改你的建站时间或者网站上线时间 
      now.setTime(now.getTime()+250); 
      days = (now - grt ) / 1000 / 60 / 60 / 24; dnum = Math.floor(days); 
      hours = (now - grt ) / 1000 / 60 / 60 - (24 * dnum); hnum = Math.floor(hours); 
      if(String(hnum).length ==1 ){hnum = "0" + hnum;} minutes = (now - grt ) / 1000 /60 - (24 * 60 * dnum) - (60 * hnum); 
      mnum = Math.floor(minutes); if(String(mnum).length ==1 ){mnum = "0" + mnum;} 
      seconds = (now - grt ) / 1000 - (24 * 60 * 60 * dnum) - (60 * 60 * hnum) - (60 * mnum); 
      snum = Math.round(seconds); if(String(snum).length ==1 ){snum = "0" + snum;} 
      document.getElementById("timeDate").innerHTML = "本站已安全运行 "+dnum+" 天 "; 
      document.getElementById("times").innerHTML = hnum + " 小时 " + mnum + " 分 " + snum + " 秒"; 
    } 
    setInterval("createtime()",250);
  </script>
  ```

#### 背景显示飘动的彩带

- 同上layout：

  ```html
  <script src="https://cdn.jsdelivr.net/gh/wallleap/cdn/js/piao.js"></script>
  ```

- 背景显示分子状线条

  ```html
  <script src="https://cdn.jsdelivr.net/gh/wallleap/cdn/js/canvas-nest.min.js"></script>
  ```

#### 樱花飘落或雪花飘落

- 樱花src：_https://cdn.jsdelivr.net/gh/wallleap/cdn/js/sakura.js_同上位置
- 雪花src: _https://cdn.jsdelivr.net/gh/wallleap/cdn/js/xuehuapiaoluo.js_或者_https://cdn.jsdelivr.net/gh/wallleap/cdn/js/snow.js_选择一个

#### 禁用按键

- 禁用了右键F12等

  ```html
  <script src="https://cdn.jsdelivr.net/gh/wallleap/cdn/js/noSomeKey.js"></script>
  ```

#### wordcout

- 在主题有字段post_wordcout：需要先npm下载hexo-wordcout才能用

- 另外默认没有单位，修改themes\next\layout\_macro\post.swig：例如

  ```html
  <span title="{{ __('post.min2read') }}">
     {{ min2read(post.content) }} 分钟
   </span>
  ```

  

#### 更多特效

- [播放器头像旋转等](<https://blog.csdn.net/u011475210/article/details/79023429#comments>)、[next深度定制](<https://blog.csdn.net/weixin_43738731/article/details/85843474>)、添加文章阅读量统计功能（next文档第三方服务部分：记得手动加Counter类）、数据统计与分析

#### 解决时延问题

![需要根据public结构](https://kivid.github.io/blog/image/解决时延.png)

- 发布后发现加载过慢于是看看是什么原因。第一个是font即谷歌字体的镜像网站无法访问，本来查了些镜像站发现不会用，而原本默认的居然能访问于是在主题配置文件中更改font字段下的host。
- 第二个是disq第三方评论网站被墙，只需将其字段设false。
- 还有个是head的背景图片url失效去主题博主网站[下载了](https://notes.doublemine.me/images/header-background.jpg)（不知道为什么能Get到可能是本地的）。但本地应用由于hexo生成public文件夹发布会找不到于是使用其url（在head-style文件）
- Aplayer是js没有找到，查看博客的repo文件。
- 网易云外链在博客只能点击几首歌，在这里发现是外链问题。依次对单曲产生外链发现是版权问题。

## 总结

- 搭建hexo博客已经是一搜一大片。如果是第一次做先弄清楚前端相关知识，再从hexo理解和文档入手。配置好环境条件：能生成一个本地能访问的前端。再者就是配置基础项可见文档，如何发布文档和上面已经说明。只是发文到此可止。

- 美化博客就像DIY（像我们这种后端）简直就是东拼西凑。第一步是换个生态好的，稳定维护的主题，我用的是魔改的Next。文档之全面让我们与代码分离开来。下面就是本博客的主要得分。

1. 先是博客插件。第一个就是给静态博客添加评论系统，参考文档发现next加入了诸多第三方评论系统。例如valine、gitment、来必力等。使用方法都很简单，注册拿到appid和appkey 插入到主题配置文件的相应字段。

2. 还有就是文章字数显示浏览次数用的wordcount和leancloud都在文档有接口

3. 另外就是参考diygod博客的小交互，例如点击烟花，吸底apalyer，背景点击变化

4. 还有些tips：摘要分割（需改主题配置），博客边界阴影，看板娘，评论区加入每日一词和正则检验，文章加密，side：头像转动，划入显示链接。

   PS：点击菜单链接中有%20应该是空格字符转义，删除||前面空格可用。菜单中除了about和404需要写其他都能生成。

5. 现在还有个点是valine admin没有部署（评论邮件通知），因为leancloud没配置好，可能要另外域名来绑定才行.