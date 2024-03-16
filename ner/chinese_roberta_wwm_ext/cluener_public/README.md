
# 语料下载地址：https://www.cluebenchmarks.com/introduce.html
cluener_public是人工标注的数据，需转成模型能识别的格式，通过text_2_bio.py进行转换

# 下载教程
细粒度命名实体识别语料下载图.png
语料文件cluener_public-效果图.png

下载完成后，将所有文件解压到当前目录

# 语料说明
CLUENER 细粒度命名实体识别

数据分为10个标签类别，分别为: 
地址（address），
书名（book），
公司（company），
游戏（game），
政府（goverment），
电影（movie），
姓名（name），
组织机构（organization），
职位（position），
景点（scene）

数据详细介绍、基线模型和效果测评，见 https://github.com/CLUEbenchmark/CLUENER

技术讨论或问题，请项目中提issue或PR，或发送电子邮件到 ChineseGLUE@163.com

测试集上SOTA效果见榜单：www.CLUEbenchmark.com