
import json
import glob
from utils.faiss import Myfaiss
from utils.query_processing import Translation
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
import polars as pl
from functools import reduce
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# http://0.0.0.0:5001/home?index=0

# app = Flask(__name__, template_folder='templates', static_folder='static')

app = Flask(__name__, template_folder='templates')

####### CONFIG #########
json_path = 'full_path.json'
bin_file = 'full_faiss.bin'

with open(json_path) as json_file:
    json_dict = json.load(json_file)

DictImagePath = {}
for key, value in json_dict.items():
    DictImagePath[int(key)] = value
MAX_ID = len(DictImagePath)

LenDictPath = len(DictImagePath)
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")

dataframe_path = "dataframe_Lxx.csv"
pldf = pl.read_csv(dataframe_path)
########################


@app.route('/home')
@app.route('/')
def thumbnailimg():
    print("load_iddoc")

    pagefile = []
    index = int(request.args.get('index'))
    if index == None:
        index = 0

    imgperindex = 100

    # imgpath = request.args.get('imgpath') + "/"
    
    page_filelist = []
    list_idx = []

    if LenDictPath-1 > index+imgperindex:
        first_index = index * imgperindex
        last_index = index*imgperindex + imgperindex

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])
            list_idx.append(tmp_index)
            tmp_index += 1
    else:
        first_index = index * imgperindex
        last_index = LenDictPath

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])
            list_idx.append(tmp_index)
            tmp_index += 1

    pagefile = []
    for imgpath, id in zip(page_filelist, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': id})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}

    return render_template('home.html', data=data)


@app.route('/imgsearch')
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=1000)

    imgperindex = 100

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}

    return render_template('home.html', data=data)


@app.route('/showsegment')
def showsegment():
    print("show segment")
    pagefile = []
    id_query = int(request.args.get('imgid'))

    imgperindex = 100
    neighbor_number = 50
    start_id    = 0         if id_query - neighbor_number <= 0      else id_query - neighbor_number
    end_id      = MAX_ID    if id_query + neighbor_number >= MAX_ID else id_query + neighbor_number

    for id in range(start_id, end_id):
        pagefile.append({'imgpath': DictImagePath[id], 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}

    return render_template('home.html', data=data)


@app.route('/textsearch')
def text_search():
    print("text search")

    pagefile = []
    text_query = request.args.get('textquery')
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=1000)

    imgperindex = 100

    print("len(list_ids)----------------", len(list_ids))

    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}

    return render_template('home.html', data=data)


@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    # fpath = fpath
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jpg")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280, 720))

    # print(img.shape)
    img = cv2.rectangle(img, [0, 0], [1200, 110], color=(255, 255, 255), thickness=-1)
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                      2.5, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/submitOD', methods=['POST'])
def submit():
    detection_data = request.json
    print("Detection Filter")
    pagefile = rerank(clip_filter=detection_data['pagefile'], detection_query=detection_data['detection_query'])
    print(detection_data['detection_query'])
    data = dict()
    data['num_page'] = detection_data['num_page']
    data['pagefile'] = pagefile
    return jsonify(data)


def filter_df(df, conditions):
    if not conditions:
        # If no conditions provided, return the original DataFrame
        return df
    else:
        # Combine conditions using logical AND
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition

        # Filter the DataFrame
        filtered_df = df.filter(combined_condition)
        return filtered_df
    

def and_operation(bool_list):
    return reduce((lambda x, y: x & y), bool_list)


def not_and_operation(bool_list):
    return not reduce((lambda x, y:  not x & y), bool_list)


def gen_np_from_df(polar_df):
    df = polar_df.select(['image_path', 'id'])
    filters = [{'imgpath': x[0], 'id': x[1]} for x in list(df.iter_rows())]
    return np.array(filters)


def rerank(clip_filter, detection_query):
    # init 
    clip_filter_pl = pl.DataFrame(clip_filter).rename({"imgpath": "image_path"})
    polar_df = pldf.join(clip_filter_pl, on='image_path')

    # filter
    conditions = [polar_df[x] >= detection_query[x]  for x in detection_query.keys()]
    
    filter_pl = filter_df(polar_df, conditions)
    filter_pl = filter_pl.sort(filter_pl.columns)

    filter_np = gen_np_from_df(filter_pl).tolist()
    return filter_np

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
