from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import json
import random

import mysql
import Yolo
app = Flask(__name__)

# 初始化数据库
sqldb = mysql.mysql()
# 初始化模型
model = Yolo.Yolov5ONNX("./archi.onnx")
print("model loaded")
# 读取模型推理出结果ID -> UUID，name
id_to_uuid = {}
with open("id_to_uuid.json", "r") as f:
    id_to_uuid = f.read()
    id_to_uuid = json.loads(id_to_uuid)
    f.close()


@app.route("/config", methods=["GET"])
def get_config():
    ret = {
        "data": {
            "imageUrl": random.choice(["/static/cover/" + str(i) + ".png" for i in range(1, 21)]),
            "color": [
                "red",
                "blue",
                "yellow",
                "green",
                "purple",
                "orange"
            ] * 3  # 前端应该用循环数组的，这是临时的解决方案

        },
        "code": 200,
        "msg": ""
    }
    return ret


@app.route("/uploadImage", methods=["POST"])
def uploadImage():
    # 解析form-data
    if (len(request.files) == 0):
        return ({"data": {}, "code": 400, "msg": "image is required_1"}, 400)
    file = request.files.get("image")
    if file is None:
        return ({"data": {}, "code": 400, "msg": "image is required_2"}, 400)

    # 将Img转换为cv2格式
    img = Image.open(file.stream)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # get size
    x_ortginal, y_original, _ = img.shape

    # 模型推理
    output, or_img = model.inference(img)
    outbox = Yolo.filter_box(output, 0.5, 0.5)

    ret_archi = []  # 返回的archi列表
    for box in outbox:
        x1, y1, x2, y2, score, archi_id = box
        x_after_resize = (x1 + x2) / 2
        y_after_resize = (y1 + y2) / 2
        # 取框的中心点作为返回值

        x = x_ortginal * x_after_resize / 640
        y = y_original * y_after_resize / 640

        percentage_x = x_after_resize / 640
        percentage_y = y_after_resize / 640

        archi_id = str(int(archi_id))
        uuid = id_to_uuid[archi_id]["uuid"]
        name = id_to_uuid[archi_id]["name"]
        ret_archi.append({
            "x": x,
            "y": y,
            "uuid": uuid,
            "name": name,
            "percentage_x": percentage_x,
            "percentage_y": percentage_y
        })

    ret = {
        "data": {
            "imageUrl": "N/A",
            "archi": ret_archi
        },
        "code": 200,
        "msg": ""
    }

    return (ret, 200)


# GET /article?id=
@ app.route('/article', methods=['GET'])
def get_article():
    article_id = request.args.get('id')  # 获取id
    if article_id is None:
        return ({"data": {}, "code": 400, "msg": "id is required"}, 400)

    sql = "SELECT * FROM archi_element WHERE id = \"" + article_id + "\""
    res = sqldb.query(sql)
    if len(res) == 0:
        return ({"data": {article_id: article_id}, "code": 404, "msg": "article not found"}, 404)
    id = res[0][0]
    logo_path = res[0][1]
    name = res[0][2]
    descri = res[0][3]
    image_path = res[0][4]
    logo_count = res[0][5]
    logo_count = random.randint(0, logo_count - 1)
    logo_path = logo_path.split(".")[0] + "_" + str(logo_count) + ".png"

    l = [
        {
            "type": "id",
            "content": id
        },
        {
            "type": "logo_path",
            "content": logo_path
        },
        {
            "type": "name",
            "content": name
        },
        {
            "type": "descri",
            "content": descri
        },
        {
            "type": "image_path",
            "content": image_path
        }
    ]
    data = {
        "id": id,
        "logo_path": logo_path,
        "name": name,
        "descri": descri,
        "image_path": image_path
    }
    ret = {"data": data,
           "code": 200,
           "msg": ""
           }
    return (ret, 200)

# static file


@ app.route('/static/<path:path>')
def get_static(path):
    return app.send_static_file(path)


if __name__ == "__main__":
    from waitress import serve
    print("server starting")
    serve(app, host="0.0.0.0", port=5000)   #本地启动服务，端口5000
