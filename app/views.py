# Basic imports
import os
import shutil
import cv2
import csv
import pandas as pd

# Flask releated imports
from flask import jsonify, request, Response, send_from_directory
from werkzeug.utils import secure_filename
import requests

# App and ML imports
from app import app, FD_model, FED_model, c_model, facenet
from app.aux_code import *

#  -------------------------------------------------------  Queries routes  ------------------------------------------------------
@app.route("/home", methods=["GET"])
def home():
    # Getting collections
    collections = os.listdir(app.config["COLLECTIONS"])
    num_collections = len(collections)

    num_analysis = 0
    aux_storage = 0
    for collection in collections:
        analysis = os.listdir(app.config["COLLECTIONS"] + collection)
        num_analysis += len(analysis)

        current_path = app.config["COLLECTIONS"] + collection + "/"
        for filename in analysis:
            file_path = current_path + filename + "/" + filename + ".mp4"
            aux_storage += os.path.getsize(file_path)
    
    size_scale = ["B","KB","MB","GB","TB"]
    aux_index = 0
    while(aux_storage > 1024 and aux_index < len(size_scale)):
        aux_storage /= 1024
        aux_index += 1
            
    aux_storage = round(aux_storage, 2)
    storage = str(aux_storage) + " " + size_scale[aux_index]

    num_uploads = len(os.listdir(app.config["UPLOADS"]))

    res = {
        "collections": num_collections,
        "analysis": num_analysis,
        "storage": storage,
        "uploads": num_uploads
    }
    
    return jsonify(res)

@app.route("/get_folders", methods=["GET"])
def get_collections():
    if request.method == "GET":
        os_list = os.listdir(app.config["COLLECTIONS"])
        final_list = []

        for item in os_list:
            if len(item.split(".")) == 1:
                final_list.append(item)

        return jsonify(final_list)
    
    return Response("Incorrect request", status=405)

@app.route("/get_items", methods=["GET"])
def get_items():
    if request.method == "GET":
        os_list = os.listdir(app.config["COLLECTIONS"] + request.args.get("filename"))
        final_list = []

        for item in os_list:
            final_list.append(item + ".mp4")

        return jsonify(final_list)
    
    return Response("Incorrect request", status=405)

@app.route("/get_analysis", methods=["GET"])
def get_analysis():
    if request.method == "GET":
        # Getting request parameters
        filename = request.args.get("filename")
        base_name = filename.split(".")[0]
        folder = request.args.get("folder")

        # Preparing variables
        base_path = app.config["COLLECTIONS"] + folder + "/" + base_name + "/" + base_name + "_"
        results = {
            "file": filename,
            "datasets": []
        }

        # First loop condition
        counter = 0
        path = base_path + str(counter) + ".csv"

        # Loop
        while(os.path.exists(path)):
            results["datasets"].append({
                "exp": [[],[],[],[],[],[],[]],
                "val": [],
                "aro": [],
                "labels": []
            })

            df = pd.read_csv(path)
            data = df.values
            
            for i, row in enumerate(data):
                for j in range(7):
                    results["datasets"][counter]["exp"][j].append(row[j])

                results["datasets"][counter]["aro"].append(row[7])
                results["datasets"][counter]["val"].append(row[8])
                results["datasets"][counter]["labels"].append(i)

            counter += 1
            path = base_path + str(counter) + ".csv"

        res = [results]
        return jsonify(res)

@app.route("/get_video", methods=["GET"])
def get_video():
    if request.method == "GET":
        filename = request.args.get("filename")
        base_name = filename.split(".")[0]
        folder = request.args.get("folder")
        
        path = app.config["COLLECTIONS"] + folder + "/" + base_name + "/"

        return send_from_directory(
            path, filename=filename, as_attachment=False
        )


#  -------------------------------------------------------  Upload routes  -------------------------------------------------------
@app.route("/get_uploads", methods=["GET"])
def get_uploads():
    files = os.listdir(app.config["UPLOADS"])
    uploads = []
    percentages = []
    counter = 0
    for filename in files:
        if len(filename.split(".")) > 1:
            # Computing file size
            aux_size = os.path.getsize(app.config["UPLOADS"] + filename)
            size_scale = ["B","KB","MB","GB","TB"]
            aux_index = 0
            while(aux_size > 1024 and aux_index < len(size_scale)):
                aux_size /= 1024
                aux_index += 1
            
            aux_size = round(aux_size,2)
            size = str(aux_size) + size_scale[aux_index]
            
            # Sending file in the same format used in frontend
            uploads.append({
                "index":    counter,
                "filename": filename,
                "size":     size,
                "format":   filename.split(".")[1]
            })
            percentages.append(100)
            counter += 1
    
    res = {
        "uploads": uploads,
        "percentages": percentages
    }

    return jsonify(res)

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if request.method == "POST":

        if request.files:
            video = request.files["file"]
            filename = video.filename

            # Handling case of unnamed file
            if filename == "":
                return Response("File must have a name", status=400)

            # Handling correct extensions
            if not allowed_extension(filename):
                return Response("Wrong file extension", status=400)

            # Handling harmful file names
            new_filename = secure_filename(filename)

            target_path = app.config["UPLOADS"] + new_filename

            if os.path.exists(target_path):
                return Response("File with the same name already exists", status=400)
                
            video.save(target_path)

            response_body = jsonify(
                { "filename": filename, "filePath": target_path}
            )
            return response_body
    
    return Response("Incorrect request", status=405)

@app.route("/delete_video", methods=["POST"])
def delete_file():
    if request.method == "POST":
        if request.form:
            path = app.config["UPLOADS"] + request.form.get("filename")
            
            if os.path.exists(path):
                os.remove(path)
                return Response("File deleted", status=200)
            else:
                return Response("File does not exists", status=404)

    return Response("Incorrect request", status=405)


#  -----------------------------------------------------  Collection routes  -----------------------------------------------------
@app.route("/create_collection", methods=["GET"])
def create_collection():
    new_name = request.args.get("new_name")
        
    try:
        os.mkdir(app.config["COLLECTIONS"] + new_name)
    except:
        return Response("Collection already exists!", status=405)
        
    os_list = os.listdir(app.config["COLLECTIONS"])
    final_list = []

    for item in os_list:
        if len(item.split(".")) == 1:
            final_list.append(item)

    return jsonify(final_list)

@app.route("/delete_collection", methods=["GET"])
def delete_collection():
    collection_name = request.args.get("collection_name")
    collection_path = app.config["COLLECTIONS"] + collection_name
        
    if os.path.exists(collection_path):
        shutil.rmtree(collection_path)
    else:
        return Response("Collection does not exist!", status=405)
        
    os_list = os.listdir(app.config["COLLECTIONS"])
    final_list = []

    for item in os_list:
        if len(item.split(".")) == 1:
            final_list.append(item)

    return jsonify(final_list)

@app.route("/delete_analysis", methods=["GET"])
def delete_analysis():
    filename = request.args.get("file_name").split(".")[0]
    collection = request.args.get("collection")

    file_path = app.config["COLLECTIONS"] + collection + "/" + filename
        
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    else:
        return Response("Collection does not exist!", status=405)
        
    os_list = os.listdir(app.config["COLLECTIONS"] + collection + "/")
    final_list = []

    for item in os_list:
        final_list.append(item + ".mp4")
        

    return jsonify(final_list)


#  ------------------------------------------------------  Analysis routes  ------------------------------------------------------
@app.route("/get_blocks", methods=["GET"])
def get_blocks():
    if request.method == "GET":
        filename = request.args.get("filename")
        folder = request.args.get("folder")
        
        # Getting total frames
        cap = cv2.VideoCapture(app.config["UPLOADS"] + filename)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap = cap.release()

        # Constants
        ms_per_frames = 600 * 4 / fps
        request_ms = 120000

        blocks = int((frames*ms_per_frames) / (request_ms))

        # Extra frames if needed
        if (frames*ms_per_frames) % (request_ms) != 0: 
            blocks += 1

        new_dir = app.config["COLLECTIONS"] + folder + "/" + filename.split(".")[0]
        os.mkdir(new_dir)

        os.rename(
            app.config["UPLOADS"] + filename, 
            new_dir + "/" + filename
        )
        return jsonify(blocks)

    return Response("Incorrect request", status=405)

@app.route("/get_predictions", methods=["GET"])
def get_predictions():
    if request.method == "GET":
        # Getting request data
        filename = request.args.get("filename")                                     # string
        base_filename = filename.split(".")[0]
        folder = request.args.get("folder")                                         # string
        block_id = int(request.args.get("block_id"))                                # int
        total_blocks = int(request.args.get("total_blocks"))                        # int
        root_dir = app.config["COLLECTIONS"] + folder + "/" + base_filename + "/"   # string

        # Getting data from the video
        cap = cv2.VideoCapture(root_dir + filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Aux variables
        anchors = get_anchors(filename)
        start, end, results = aux_variables(block_id, total_blocks, frames, fps, anchors)
        counter, dif, num_analysis = 0, 0, 1

        # Analysis itself
        carry_on, frame = cap.read()
        while (carry_on and counter < end):
            aux_dif = (counter + dif) - (fps*num_analysis*0.6)
 
            if aux_dif > 0:
                if (counter >= start):
                    analysis, anchors = analize_frame(FD_model, FED_model, c_model, facenet, frame, anchors, root_dir)
                    results = merge_results(results, analysis, num_analysis)

                dif = aux_dif
                num_analysis += 1

            counter += 1
            carry_on, frame = cap.read()

        # Saving analysis inside csv file
        csv_path = root_dir + base_filename
        write_csv(csv_path, results)
        save_anchors(filename, anchors)

        if block_id + 1 == total_blocks:
            results = final_results(csv_path, anchors)

            # Cleaning temp file
            os.remove(app.config["UPLOADS"] + base_filename + ".csv")

        return jsonify(results)

    return Response("Incorrect request", status=405)


