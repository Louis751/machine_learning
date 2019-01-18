#!/usr/bin/evn python
# -*- coding:utf-8 -*-
# author:stealth

list = {
    "植物":
        {"草本植物":
             ["牵牛花", "瓜叶菊", "翠菊", "冬小麦", "甜菜"],
         "水本植物":
             ["乔木", "灌木", "半灌木", "如松", "杉", "樟"],
         "水生植物":
             ["荷花", "千屈菜", "菖蒲", "黄菖蒲", "水葱", "再力花", "梭鱼草"]},
    "动物":
        {"两栖动物":
             ["山龟", "山鳖", "石蛙", "娃娃鱼", "蟾蜍", "龟", "鳄鱼", "蜥蜴", "蛇"],
         "禽类":
             ["雏鸡", "原鸡", "长命鸡", "昌国鸡", "斗鸡", "长尾鸡", "乌骨鸡"],
         "哺乳类动物":
             ["虎", "狼", "鼠", "鹿", "貂", "猴", "嫫", "树懒", "斑马", "狗"]}
}

go = True
leve1 = []
while go:
    for k, v in enumerate(list, 1):
        print(k, v)
        leve1.append(v)
    u1 = input("请输入一级分类序号或输入q退出：>>")
    if u1.isdigit() and int(u1) > 0 and int(u1) <= k:
        u1 = int(u1)
        leve2 = []
        while go:
            for k, v in enumerate(list[leve1[u1 - 1]], 1):
                print(k, v)
                leve2.append(v)

            u2 = input("请输入二级分类序号或输入b返回或输入q退出：>>")
            if u2.isdigit() and int(u2) > 0 and int(u2) <= k:
                u2 = int(u2)
                leve3 = []
                while go:
                    for i in list[leve1[u1 - 1]][leve2[u2 - 1]]:
                        print(i)
                    u3 = input("请输入b返回或输入q退出：>>")
                    if u3.isalpha():
                        if u3.lower() == 'b':
                            leve3.clear()
                            break
                        if u3.lower() == 'q':
                            go = False
                        else:
                            print("对不起您输入的有误，请输入字母b或q！")
                    else:
                        print("对不起您输入的有误，请输入字母b或q！")
            elif u2.isalpha():
                if u2.lower() == 'b':
                    leve2.clear()
                    break
                if u2.lower() == 'q':
                    go = False
                else:
                    print("对不起您输入的有误，请输入二级分类序号或字母b或字母q！")
            else:
                print("对不起您输入的有误，请输入二级分类序号或字母b或字母q！")
    elif u1.isalpha():
        if u1.lower() == 'q':
            go = False
        else:
            print("对不起您输入的有误，请输入一级分类序号或输入q退出！")
    else:
        print("对不起您输入的有误，请输入一级分类序号或输入q退出！")
