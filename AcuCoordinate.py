#_*_ coding:utf-8 _*_
import cv2
import numpy as np
import math
import time
from PIL import Image, ImageDraw, ImageFont
import types
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pymysql
from sshtunnel import SSHTunnelForwarder
font = ImageFont.truetype("/Users/taohuadao/Documents/PycharmProjects/font/ZiTiGuanJiaTianZhen-2.ttf", 20, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小

server=SSHTunnelForwarder(
ssh_address_or_host=('213f55x557.iask.in',26123),
ssh_username='lee',
ssh_password='qqqq1111',
remote_bind_address=('localhost',3306)
)
server.start()
conn = pymysql.connect(host='localhost',port=server.local_bind_port,user='lee',password='qqqq1111',db='shuowen',charset='utf8')
cursor = conn.cursor()
cursor.execute("select*from acupoint")
row_1=cursor.fetchall()
print (row_1[1])
def on_press(event):
    print("my position:" ,event.button,event.xdata, event.ydata)

#处理图片，返回凸包&轮廓的点集（defects, cnt）
def processImg(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转换为二值图
    line = img_gray[1, 1] / 4 + img_gray[img_gray.shape[0] - 1, img_gray.shape[1] - 1] / 4 + img_gray[
        1, img_gray.shape[1] - 1] / 4 + \
           img_gray[img_gray.shape[0] - 1, 1] / 4
    ret, thresh1 = cv2.threshold(img_gray, line - 20, 255, cv2.THRESH_BINARY_INV)  # 手白背景黑
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh1, kernel, iterations=1)  # 腐蚀手的轮廓
    _, contours, hierarchy = cv2.findContours(thresh, 2, 1)  # 检测轮廓
    cnt = contours[0]  # cnt为第一个轮廓
    hull = cv2.convexHull(cnt, returnPoints=False)  # 根据图像的轮廓点，通过函数 convexHull 转化成凸包的点点的坐标
    defects = cv2.convexityDefects(cnt, hull)
    return defects, cnt
#返回手臂有效的凸包，返回一个list数组，数组储存每个凸包的最远点纵坐标，起点，最远点，终点坐标
def validateConvex(defects,cnt):
    list=[]
    distanceL =[]
    count1= 0
    if defects.shape[0]:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dis = distance(start,end,far)
            list.append((far[1],start,far,end))
            distanceL.append(dis)
        min = sorted(distanceL)[defects.shape[0]-6]
        for j in range(defects.shape[0]):
            if distanceL[j]<min:
                list[j]=(0,0,0,0)
    return list
#打印手臂凸包，以描点划线的方式将检测到的有效的6个点画出来
def printConvex(img,defects,cnt,list):
    if defects.shape[0]:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            if list[i][0] != 0:
                cv2.line(img, start, end, [0, 255, 0], 2)
                cv2.circle(img, far, 5, [0, 0, 255], -1)
        showImage(img)
        return
#检测有效的点， 返回一个last的数组，里面储存手的11个关键点
def validatePoint(list):
    deleteZero(list)
    last = []
    for i in range(13):
        last.append((0,0))
    if (list[list.index(sorted(list)[1])][2][0] > list[list.index(sorted(list)[0])][2][0]): #最远点最高的点的横坐标小一点的话，即在左边
        last[2] = list[list.index(sorted(list)[1]) - 1][1]  #右右凸包的起点坐标
        last[3]  = list[list.index(sorted(list)[1]) - 1][2]  #右右凸包的最远点坐标
        last[4]  = list[list.index(sorted(list)[1])][1]      #右凸包的起点坐标
        last[5]  = list[list.index(sorted(list)[1])][2]      #右凸包的最远点坐标
        last[6]  = list[list.index(sorted(list)[0])][1]      #左凸包的起点坐标
        last[7]  = list[list.index(sorted(list)[0])][2]      #左凸包的最远点坐标
        last[8]  = list[list.index(sorted(list)[0]) + 1][1]  #左左凸包的起点
        last[9]  = list[list.index(sorted(list)[0]) + 1][2]  #左左凸包的最远点
        last[10]  = list[list.index(sorted(list)[0]) + 1][3]  #左左凸包的终点
    else:  #最高的点在右边
        last[2] = list[list.index(sorted(list)[0]) - 1][1]
        last[3] = list[list.index(sorted(list)[0]) - 1][2]
        last[4] = list[list.index(sorted(list)[0])][1]
        last[5] = list[list.index(sorted(list)[0])][2]
        last[6] = list[list.index(sorted(list)[1])][1]
        last[7] = list[list.index(sorted(list)[1])][2]
        last[8] = list[list.index(sorted(list)[1]) + 1][1]
        last[9] = list[list.index(sorted(list)[1]) + 1][2]
        last[10] = list[list.index(sorted(list)[1]) + 1][3]
    temp1 = list[list.index(sorted(list)[-2])]
    temp2 = list[list.index(sorted(list)[-1])]   #纵坐标最低的两个点
    if temp1[1][0] < temp2[1][0]:   #last[10] = 左边, last[0] = 右边
        last[11] = temp1[2]
        last[12] = temp1[3]
        last[1] = temp2[2]
        last[0] = temp2[1]
    else:
        last[1] = temp1[2]
        last[0] = temp1[1]
        last[11] = temp2[2]
        last[12] = temp2[3]
    return last
#删除list里面为零的点
def deleteZero(list):
    for i in range(len(list)):
        if list[i][0]==0:
            del list[i]
            deleteZero(list)
            break
    return list
#打印图片的函数
def showImage(img):
    #fig = plt.figure()
    #plt.imshow(img, animated=True)
    #fig.canvas.mpl_connect('button_press_event', on_press)
    #plt.show()
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#知道三个点的坐标求夹角余弦
#def cosFar(start, far, end):
#    x1= start[0]
#    y1= start[1]
#    x2= far[0]
#    y2= far[1]
#    x3= end[0]
#    y3= end[1]
#    distance1= math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
#    distance2= math.sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2))
#    distance= distance1*distance2
#    vectorM=(x3-x2)*(x1-x2)+(y3-y2)*(y1-y2)
#    cos=vectorM/distance
#    return cos
#知道三个点的坐标求最远点到直线的距离
def distance(start, far, end):
    x1 = start[0]
    y1 = start[1]
    x2 = far[0]
    y2 = far[1]
    x3 = end[0]
    y3 = end[1]
    A = y3-y1
    B = x1-x3
    C = (y1-y3)*x1+(x3-x1)*y1
    D = A*x2+B*y2+C
    if D<0:
        D= D*(-1)
    return D/math.sqrt(A*A+B*B)
#已知一个轮廓和一个点，求这个点在轮廓上的下标，若不存在于这个轮廓上则返回-1
def index(cnt,point):
    j=0
    for i in cnt:
        if i[0][0]==point[0] and i[0][1]==point[1]:
             return j
        j+=1
    return -1

#纵向：已知轮廓坐标集，cnt，起点坐标start, 终点坐标end，将此段分为m份，求起点坐标到终点坐标轮廓间每个点的坐标，返回一个坐标的数组或元组
def cntPoint(cnt, start, end, m):
    #cv2.circle(img, start, 10, [255, 0, 0], -1)
    #cv2.circle(img, end, 10, [0, 255, 0], -1)
    li=[[0,0]]
    list=li*(m+1)
    index1=index(cnt,start)
    index2=index(cnt,end)
    index3=index1
    if(index1<=index2):
        add=round(((index2-index1)/m),2)
        count=0
        while(index3<=index2):
            if [0, 0] in list:
                list.remove([0, 0])
                app = [cnt[index3][0][0], cnt[index3][0][1]]
                list.append(app)
                count=count+1
                index3=index1+int(count*add)
        if index3-count!=index2:
            if [0, 0] in list:
                list.remove([0, 0])
            else:
                list.pop()
            list.append([end[0],end[1]])
    else:
        add = round((index2 - index1) / m,2)
        count=0
        while (index3 >= index2):
            if [0, 0] in list:
                list.remove([0,0])
                app = [cnt[index3][0][0], cnt[index3][0][1]]
                list.append(app)
                count+=1
                index3 = index1+ int(count*add)
        if index3 - count != index2:
            if [0,0] in list:
                list.remove([0,0])
            else:
                list.pop()
            list.append([end[0],end[1]])
    return list
#横向：已知起点坐标，终点坐标，将此段分为n份，求线上每一个点的坐标,返回一个坐标数组或元组
def xPoint(start, end, n):
    li=[[0,0]]
    list=li*(n+1)
    x=(end[0]-start[0])/n
    y=(end[1]-start[1])/n
    mox=start[0]
    moy=start[1]
    for i in range(n+1):
        list.remove([0,0])
        list.append([round(mox,2),round(moy,2)])
        mox+=x
        moy+=y
    return list
#二维数组的确定，已知左上，右上，左下，右下坐标,横份n，纵份m，返回一个二维数组，四个点必须在轮廓上
def coordinate(cnt, a1, a2, b1, b2, m, n):
    result = [[0 for x in range(n+1)] for y in range(m+1)]
    result[0][0]=a1
    result[0][n]=a2
    result[m][0]=b1
    result[m][n]=b2
    edge1=cntPoint(cnt, a1, b1, m)
    edge2=cntPoint(cnt, a2, b2, m)
    #显示两点之间的轮廓
    #showTuplePoints(edge1,0,255,255)
    #showTuplePoints(edge2,0,255,0)
    for i in range(m+1):
        result[i][0]=edge1[i]
        result[i][n]=edge2[i]
        for j in range(n+1):
            result[i][j]=xPoint(result[i][0],result[i][n], n)[j]
    return result
#给定一个坐标数组，将轮廓上的点画出来，a,b,c是颜色rgb值
def showTuplePoints(points,a,b,c):
    for i  in points:
        x = int(i[0])
        y = int(i[1])
        z = (x, y)
        cv2.circle(img, z, 1, [a, b, c], -1)
#删除轮廓集里start和end中间的点,返回一个list
def fingerRemove(cnt,start,end):
    index1=index(cnt,start)
    index2=index(cnt,end)
    x=cnt
    if index1<index2:
        for i in range(index1+1,index2):
            x=np.delete(x,index1+1,0)
    else:
        for i in range(0,index2):
            x=np.delete(x,0,0)
        size=int(x.size/2)
        index3=index(x,start)
        for j in range(0,size-index3-1):
            x=np.delete(x,index3+1,0)
    list=x.tolist()
    rlist=insertP(list,start,end)
    return rlist
def insertP(list,start,end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]
    index1 = list.index([[x1, y1]])
    if (abs(x2 - x1) > abs(y2 - y1)):
        xnum = abs(x2 - x1)-1
        for i in range(1, xnum+1):
            addx = int(i * (x2 - x1) / abs(x2 - x1))
            addy = getNear(i * ((y2 - y1) / xnum))
            list.insert(index1 + i, [[x1 + addx, y1 + addy]])
    else:
        ynum = abs(y2 - y1)-1
        for i in range(1, ynum+1):
            addy = int(i * (y2 - y1) / abs(y2 - y1))
            addx = getNear(i * ((x2 - x1) / ynum))
            list.insert(index1 + i, [[x1 + addx, y1 + addy]])
    rlist = np.array(list)
    return rlist
def getNear(x):
    if(x>0):
        if (x-int(x)<0.5):
            return int(x)
        else:
            return int(x)+1
    else:
        if(int(x)-x<0.5):
            return int(x)
        else:
            return int(x)-1

#优化手的轮廓，删除手指
def palm(cnt,lastP):
    top1=lastP[2]
    down1=lastP[1]
    left=lastP[3]
    newp1=newP(top1,down1,left)
    top2=lastP[10]
    down2=lastP[11]
    right=lastP[9]
    newp2=newP(top2,down2,right)

    x1=index(cnt,lastP[2])
    x2=index(cnt,lastP[10])
    cnt[x1][0][0] = newp1[0]
    cnt[x1][0][1] = newp1[1]
    cnt[x2][0][0] = newp2[0]
    cnt[x2][0][1] = newp2[1]

    o1=fingerRemove(cnt,lastP[1],newp1)
    o2=fingerRemove(o1,newp1,lastP[3])
    o3=fingerRemove(o2,lastP[3],lastP[5])
    o4=fingerRemove(o3,lastP[5],lastP[7])
    o5=fingerRemove(o4,lastP[7],lastP[9])
    o6=fingerRemove(o5,lastP[9],newp2)
    o7=fingerRemove(o6,newp2,lastP[11])
    splitIndex=index(o7,lastP[0])
    r=np.split(o7,[splitIndex],axis=0)
    #for i in o7:
        #cv2.circle(img, (i[0][0], i[0][1]), 1, [0, 0, 255], -1)
    #r1=np.concatenate([r[1],r[0]],axis=0)
    return o7
def newP(p1,p2,p3):
    x1=p1[0]
    x2=p2[0]
    x3=p3[0]
    y1=p1[1]
    y2=p2[1]
    y3=p3[1]
    t=round((x2-x1)/(y2-y1),3)
    newx=int((-t*y2+t*t*x3+t*y3+x2)/(1+t*t))
    newy=int(-t*(newx-x3)+y3)
    return (newx,newy)

#给定一个轮廓，画出轮廓上的点
def showcnt(cnt):
    x=0
    for t in cnt:
        print(t)
        showTuplePoints(t,200,200,200)
    showImage(img)
#给定一个二维数组，将其储存的坐标画出来
def showCoordinate(coordinate):
    for aa in coordinate:
        for a in aa:
            x = int(a[0])
            y = int(a[1])
            z = (x, y)
            cv2.circle(img, z, 1, [0, 0, 0], -1)



img=cv2.imread('/Users/taohuadao/PycharmProjects/Acu/images/arm-left-back.jpg')
#第一步，处理图片，检测凸包和轮廓
step1=processImg(img)
#第二步，将检测到的六个关键点储存到list数组
step2=validateConvex(step1[0],step1[1])
#打印检测到的六个凸包（红点，绿线）
#printConvex(img,step1[0],step1[1],step2)
#根据六个点，储存所需的11个点，用一个数组储存坐标，逆时针。
step3=validatePoint(step2)

##改善过的手掌轮廓，起点在中指
x=palm(step1[1],step3)
#手臂
result3=coordinate(x, (60, 638), (145, 638), step3[11], step3[1], 50, 10)
#手掌
result4=coordinate(x, step3[11], step3[1], step3[7], step3[5], 15, 10)
#整条手臂
result5=coordinate(x, (60, 638), (145, 638),step3[7], step3[5],60,15)


#for i in result5:
#    print(i)
#定位穴位：
#横坐标acudb[id][3],纵坐标acudb[id][4]
#index_x=acudb[id][3],index_y=acudb[id][4]
#参数为图片，二维坐标数组，穴位序号，穴位数据库
def drawAcu(img, acuCooridinate, id, acudb):
    index_x = acudb[id][3]
    index_y = acudb[id][4]
    print(index_x)
    print(index_y)
    acuName=acudb[id][1]
    acuCo= (int(acuCooridinate[index_x][index_y][0]),int(acuCooridinate[index_x][index_y][1]))
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    draw.text(acuCo, acuName, (0, 0, 255), font=font)
    img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.circle(img, acuCo, 5, [255, 255, 0], -1)
    showImage(img)
def drawAcus(img, acuCooridinate, ids, acudb):
    for id in ids:
        drawAcu(img, acuCooridinate,id,acudb)
drawAcus(img, result5, [1,2,3], row_1)

'''
#穴位根据a[x][y]确定
def showAcu(img,name,point):
    m=0
    for j in name:
        n = 0
        for i in point[m]:
            x=(int(i[0]),int(i[1]))
            cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            draw.text(x, j[n], (0, 0, 255), font=font)
            img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            cv2.circle(img, x, 2, [255, 255, 0], -1)
            n+=1
        m+=1
    showImage(img)
def drawAcu(lr):
    handAcu = ['少商','鱼际','中冲','劳宫','少府']
    sarmAcu = ['太渊', '经渠', '列缺', '孔最', '尺泽','大陵','内关','间使','杀门','曲泽','神门','阴','通里','灵道']
    if (lr==0):#0代表右手手掌，1代表左手手掌
        handAcupoint = [ step3[2],result2[8][9],step3[6],result2[6][5],result2[6][2]]
        sarmAcupoint = [result1[0][9], result1[2][9], result1[4][9], result1[20][7], result1[30][7],result1[0][5],result1[5][5]
                        ,result1[9][5],result1[13][5],result1[30][4],result1[0][2],result1[2][2],result1[4][2],result1[5][2]]
    else:
        handAcupoint = [(step3[10][0]+5,step3[10][1]+8),result2[7][0],(step3[6][0],step3[6][1]+3),result2[6][5],result2[6][8]]
        sarmAcupoint=[result1[0][1],result1[2][1],result1[4][1],result1[20][3],result1[30][4],result1[0][5],result1[5][5]
                        ,result1[9][5],result1[13][5],result1[30][6],result1[0][8],result1[2][8],result1[4][8],result1[5][8]]
    showAcu(img,[handAcu,sarmAcu],[handAcupoint,sarmAcupoint])
#drawAcu(1)
barmAcu=['侠白','天府','天泉','青灵']
barmAcupointl=[result3[6][1],result3[7][1],result3[9][6],result3[3][8]](10,10)
barmAcupointr=[result3[6][9],result3[7][9],result3[9][4],result3[3][2]](10,10)
手背：：

handAcu=['商阳','二间','三间','合谷','少泽','前谷','后溪','腕骨']
handAcupointl=[(step3[4][0],step3[4][1]+7),(step3[5][0],step3[5][1]-4),result1[6][7],(step3[9][0],step3[9][1]-8),(step3[10][0],step3[10][1]+8]),result1[4][10],result1[5][9],result[9][9]]
handAcupointr=[(step3[8][0],step3[8][1]+7),(step3[9][0],step3[9][1]-4),result1[6][2],(step3[9][0],step3[9][1]-8),(step3[2][0],step3[2][1]+8]),result1[4][0],result1[5][1],result[9][1]]

sarmAcu=['阳溪','偏历','温溜','下廉','上廉','手三里','曲池','阳谷','养老','支正','小海']
sarmAcupointl=[result2[10][10],result2[7][9],result2[12][9],result2[22][9],result2[24][9],result2[26][9],result2[30][10],result2[0][9],result2[1][7],result2[8][10],result2[30][8]]
sarmAcupointr=[result2[0][0],result2[7][1],result2[12][1],result2[22][1],result2[24][1],result2[26][1],result2[30][0],result2[0][1],result2[1][3],result2[8][0],result2[30][2]]

barmAcu=['时','手五里','臂','肩贞']
barmAcupointl=[result3[1][9],result3[3][9],result3[10][10],result3[10][0]]
barmAcupointr=[result3[1][1],result3[3][1],result3[10][0],result3[10][10]]
'''

