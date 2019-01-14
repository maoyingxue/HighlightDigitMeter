import numpy as np
import cv2
import json
from train_test import test
import math
def digit_meter_1(image,info):
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, gray.max() - 25, gray.max(), cv2.THRESH_BINARY)
    #print(thresh.shape)
    rows = np.sum(thresh, axis=1)
    mark = 0
    total = 0
    row_points = []
    for i, row in enumerate(rows):
        if mark == 0 and row > 1000:
            start = i
            mark = 1
            total = total + row
        elif mark == 1:
            total = total + row
            if row < 1000:
                end = i
                mark = 0
                row_points.append(start)
                row_points.append(end)
                row_points.append(total)
                total = 0
    row_points = np.array(row_points, dtype="int").reshape((-1, 3))
    #print("row_points:",row_points)
    args = np.argsort(-row_points[:, 2])
    args=args[:info["nRows"]]
    args=args[np.argsort(args)]
    row_points = row_points[args]
    #print("row_points:",row_points)
    blobs = []
    for i, row_point in enumerate(row_points):
        thresh1 = thresh[row_point[0]:row_point[1]]
        cols = np.sum(thresh1, axis=0)
        mark = 0
        total = 0
        col_points = []
        for j, col in enumerate(cols):
            if mark == 0 and col > 1000:
                start = j
                mark = 1
                total = total + col
            elif mark == 1:
                total = total + col
                if col < 1000:
                    end = j
                    mark = 0
                    col_points.append(start)
                    col_points.append(end)
                    col_points.append(total)
                    total = 0
        col_points = np.array(col_points, dtype="int").reshape((-1, 3))
        print(col_points)
        for j, col_point in enumerate(col_points):
            img = thresh[row_point[0] - 2:row_point[1] + 2, col_point[0] - 2:col_point[1] + 2]
            if img.shape[0] > img.shape[1]:
                new_img = np.zeros([img.shape[0], img.shape[0]])
                subs = img.shape[0] - img.shape[1]
                subs = subs // 2
                new_img[:, subs:subs + img.shape[1]] = img
                new_img = cv2.resize(new_img, (28, 28))
                new_img = new_img.reshape(-1, 784)
                new_img = np.minimum(new_img, 1)
                predict=test(new_img)
                col_points[j][2]=predict
        #print(col_points)
        col_points=col_points.tolist()
        new_col_points=[col_point for col_point in col_points if col_point[2]!=10]
        #print(new_col_points)
        blobs.append(new_col_points)
    print(blobs)
    blobs=np.array(blobs)
    #print(blobs.shape)
    values=[]
    for i,blob in enumerate(blobs):
        vals=[]
        nCols=info["row"+str(i+1)]["nCols"]
        print(blob)
        if nCols>1:
            distances=[]
            for j in range(1,blob.shape[0]):
                dis=blob[j][0]-blob[j-1][1]
                distances.append(dis)
            distances=np.array(distances)
            indexs=np.argsort(-distances)
            index=indexs[:nCols-1]
            index=index[np.argsort(index)]
            index=index+1
            index=np.insert(index,0,0)
            index=np.append(index,len(blob))
            print(index)
            for j in range(len(index)-1):
                nDecimals=info["row"+str(i+1)]["col"+str(j+1)]["nDecimals"]
                t_b=blob[index[j]:index[j+1]]
               # print(t_b)
                tmp=""
                for k in range(t_b.shape[0]-1,-1,-1):
                    #print(k)
                    tmp=tmp+str(blob[blob.shape[0]-k-1][2])
                    if k==nDecimals:
                        tmp=tmp+"."
                val=float(tmp)
                val = round(val, nDecimals)
                vals.append(val)
        else:
            nDecimals = info["row" + str(i + 1)]["col1"]["nDecimals"]
            tmp=""
            for j in range(blob.shape[0]-1,-1,-1):
                tmp=tmp+str(blob[blob.shape[0]-j-1][2])
                if j==nDecimals:
                    tmp=tmp+"."
            val=float(tmp)
            val=round(val,nDecimals)
            vals.append(val)
        values.append(vals)
    return values
if __name__ == '__main__':
    file = open("config/" + "number5_1.json")
    info = json.load(file)
    img=cv2.imread("template/number5_1.jpg")
    values=digit_meter_1(img,info)
    print(values)