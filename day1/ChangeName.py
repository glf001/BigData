#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.7.13

import os
#获取目录下文件名清单
files = os.listdir("C:\\Users\\GUOLONGFEI\\Desktop\\test")
#对文件名清单里的每一个文件名进行处理
for filename in files:
    portion = os.path.splitext(filename)#portion为名称和后缀分离后的列表
    if portion[1] ==".csv":
        newname = portion[0]+".xlsx"#要改的新后缀#改好的新名字
        print(filename)
        os.chdir("C:\\Users\\GUOLONGFEI\\Desktop\\test")#修改工作路径
        os.rename(filename,newname)


