# 仓库文件说明

- /static/image/ : 所有 avchi 对应实景图图片
- /static/logo/ : 所有 avchi 对应图片

- ./id_to_uuid.json : id 到 {uuid,name} 的映射
- ./table.sql : mysql 数据库表结构 mysql 主要存小文章的
- ./archi.onnx : Yolo 模型

- ./mysql.py : mysql 数据库操作
- ./server.py : flask 服务端代码
- ./Yolo.py : Yolo 模型代码

# API 说明

## GET /config

### 请求参数

无

### 返回参数

```json
{
  "data": {
    "imageUrl": [],
    "color": {}
  },
  "code": 200,
  "msg": ""
}
```

data.imageUrl: 首页背景图片

data.color: 首页背景颜色

code: 状态码

msg: 状态信息

## POST /uploadImage

### 请求参数

请求体为 form-data，有一个参数为 image，值为图片文件

### 返回参数

```json
{
  "code": 200,
  "data": {
    "arhci": [
      {
        "name": "广州塔",
        "uuid": "bb2e30f1-0a71-414b-b2f6-870877452c67",
        "x": 251.1353759765625,
        "y": 436.5895690917969,
        "percentage_x": 0.3923990249633789,
        "percentage_y": 0.6821712017059326
      }
    ],
    "imageUrl": "N/A"
  },
  "msg": ""
}
```

data.arhci: 识别出的建筑物信息

data.arhci.name: 建筑物名称

data.arhci.uuid: 建筑物 uuid

data.arhci.x: 建筑物在图片上的 x 坐标（中心值）

data.arhci.y: 建筑物在图片上的 y 坐标（中心值）

data.arhci.percentage_x: 建筑物在图片上的 x 坐标（百分比）

data.arhci.percentage_y: 建筑物在图片上的 y 坐标（百分比）

data.imageUrl: 上传的图片地址（没有用）

code: 状态码

msg: 状态信息

## GET /article

### 请求参数

Params:
id : uuid

### 返回参数

```json
{
  "code": 200,
  "data": {
    "id": "bb2e30f1-0a71-414b-b2f6-870877452c67",
    "logo_path": "xxx",
    "name": "广州塔",
    "descri": "广州塔，位于广州市海珠区赤岗塔附近，是一座塔状建筑，是广州市的标志性建筑之一。",
    "image_path": "xxx"
  },
  "msg": ""
}
```

data:

- id: uuid
- logo_path: logo 图片地址
- name: 建筑物名称
- descri: 建筑物描述
- image_path: 实景图图片地址

code: 状态码

msg: 状态信息
