from app import app

# Analysis' imports
import os
import pandas as pd
import csv
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image as Image
import cv2

def allowed_extension(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]
    
    if not ext.upper() in app.config["ALLOWED_EXTENSIONS"]:
        return False

    return True

# -----------------------------------------------   Pre-analysis code   -----------------------------------------------   
def get_anchors(filename):
    anchors_csv = app.config["UPLOADS"] + filename.split(".")[0] + ".csv"
    anchors = {
        "threshold": 1.6,
        "faces": []
    }

    if os.path.exists(anchors_csv):
        with open(anchors_csv) as csv_file:
            aux_anchors = [line.split(",") for line in csv_file]
            
            for anchor in aux_anchors:
                # String -> Float
                for i in range(len(anchor)):
                    anchor[i] = float(anchor[i])

                tensor = torch.tensor(anchor).type(torch.FloatTensor)
                tensor = tensor.view((1,512))
                anchors["faces"].append(tensor)

    return anchors

def aux_variables(block_id, total_blocks, frames, fps, anchors):
    start = int((frames * block_id) / total_blocks)
    end = int((frames * (block_id + 1)) / total_blocks)

    results = []

    for _ in range(len(anchors["faces"])):
        results.append({
            "exp": [[],[],[],[],[],[],[]],
            "val": [],
            "aro": []
        })
    
    return start, end, results


# -----------------------------------------------   Aux-analysis code   -----------------------------------------------
def calc_margin(bx,by,bw,bh):
    margin = 10

    bx -= int(margin)
    by -= int(margin)

    if bx < 0:
        bx = 0
        bw += int(margin)
    else :
        bw += margin * 2

    if by < 0:
        by = 0
        bh += int(margin)
    else:
        bh += margin * 2

    return bx,by,bw,bh

def merge_results(dataset, analysis, num_analysis):
    
    # Adding new face
    while len(dataset) < len(analysis):
        dataset.append({
            "exp": [[],[],[],[],[],[],[]],
            "val": [],
            "aro": []
        })

        # Filling empty space
        for i in range(num_analysis):
            for j in range(7):
                dataset[-1]["exp"][j].append(0)

            dataset[-1]["aro"].append(0)
            dataset[-1]["val"].append(0)

    # Storing analysis
    for k, results in enumerate(analysis):
        for l in range(7):
            dataset[k]["exp"][l].append(results["exp"][l])
                            
        dataset[k]["aro"].append((results["aro"]))
        dataset[k]["val"].append((results["val"]))

    return dataset

def iou(box1,box2):
    if box1[0] <= box2[0]:
        x1,y1,w1,h1 = box1
        x2,y2,w2,h2 = box2
    else:
        x1,y1,w1,h1 = box2
        x2,y2,w2,h2 = box1

    # Condition for interception
    if x1 + w1 > x2 and y1 + h1 > y2:
        if x1 + w1 >= x2 + w2:
            inter_width = w2
        else:
            inter_width = x1 + w1 - x2


        if y1 <= y2:
            if y1 + h1 >= y2 + h2:
                inter_height = h2
            else:
                inter_height = y1 + h1 - y2
        else:
            if y2 + h2 >= y1 + h1:
                inter_height = h1
            else:
                inter_height = y2 + h2 - y1

        inter_area = inter_height * inter_width

    else:
        inter_area = 0

    area1 = (w1 * h1)
    area2 = (w2 * h2)

    if area1 > area2:
        iou_value = inter_area / area2
    else:
        iou_value = inter_area / area1

    return iou_value



# ---------------------------------------------   Machine Learning stuff   --------------------------------------------
def face_recognition(facenet, anchors, sample, root_dir, analized):

    # Getting 512x1 vector 
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(sample)
    tensor = tensor.view(1,3,112,112)

    with torch.no_grad():
        face = facenet(tensor)

    # Face recognition
    face_id = None
    distance = anchors["threshold"]

    for i, anchor in enumerate(anchors["faces"]):
        anchor_dist = torch.dist(anchor, face)

        if anchor_dist <= distance:
            face_id = i
            distance = anchor_dist

    print(len(analized))
    print(face_id)
    if face_id is None or analized[face_id] == True:
        cv2.imwrite(root_dir + "face" + str(len(anchors["faces"])) + ".jpg", sample)
        anchors["faces"].append(face)
        analized.append(True)
        face_id = None
    else:
       analized[face_id] = True

    return face_id, anchors, analized

def affect_computing(FED_model, c_model, sample):
    sample = Image.fromarray(sample).convert("L")
        
    # Getting tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.486], std=[0.226])
    ])
    tensor = transform(sample)
    tensor = tensor.view(1,1,112,112)

    # Finding expressions
    with torch.no_grad():
        output1, values = FED_model(tensor)
        output2 = c_model(values)[0]

    return F.softmax(output1, dim=1)[0], output2


#  --------------------------------------------------   Main method  --------------------------------------------------
def analize_frame(FD_model, FED_model, c_model, facenet, frame, anchors, root_dir):
    # Preparing results' format
    results = []

    analized = []
    for anchor in anchors["faces"]:
        analized.append(False)

    for face in anchors["faces"]:
        results.append({
            "exp": [0,0,0,0,0,0,0],
            "val": 0,
            "aro": 0
        })

    # Getting faces
    boxes = FD_model.detectMultiScale(frame)

    if len(boxes) > 1:
        # Let's make sure the same person was not detected twice
        for i in range(len(boxes)):
            box1 = boxes[i]

            # Making sure that face was not erase
            if box1[2] > 0:
                for j in range(i+1, len(boxes)):
                    box2 = boxes[j]
                    
                    # Making sure that face was not erase pt2
                    if box2[2] > 0:
                        # If IoU is too high, it's the same person
                        if iou(box1, box2) > 0.5:
                            # Taking the smallest detection
                            if box1[2]*box1[3] >= box2[2]*box2[3]:
                                boxes[i] = np.zeros((4,))
                                break
                            else:
                                boxes[j] = np.zeros((4,))
                            
        # Selecting remaining boxes
        final_boxes = []
        for box in boxes:
            if box[2] > 0:
                final_boxes.append(box)
            
        boxes = final_boxes

    
    for bx,by,bw,bh in boxes:
        # Taking coordinates and wrapping extra pixels
        # bx,by,bw,bh = calc_margin(bx,by,bw,bh)

        # Extracting the face and getting gray
        sample = frame[by:by+bh, bx:bx+bw]
        sample = cv2.resize(sample, dsize=(112, 112), interpolation = cv2.INTER_AREA)
        
        # ML computing
        face_id, anchors, analized = face_recognition(facenet, anchors, sample, root_dir, analized)
        exp, cont_var = affect_computing(FED_model, c_model, sample)

        # Storing results
        if face_id is None:
            face_id = len(results)
            results.append({
                "exp": [0,0,0,0,0,0,0],
                "val": 0,
                "aro": 0
            })

        for i in range(7):
            results[face_id]["exp"][i] = 100*round(exp[i].item(), 2)
            
        results[face_id]["aro"] = round(cont_var[0].item(), 3)
        results[face_id]["val"] = round(cont_var[1].item(), 3)

    return results, anchors
#  --------------------------------------------------------------------------------------------------------------------


# Post-analysis code
def write_csv(csv_path, results):
    for i, dataset in enumerate(results):

        with open(csv_path + "_" + str(i) + ".csv", 'a', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(dataset["aro"])):
                row = []
                for j in range(7):
                    row.append(dataset["exp"][j][i])

                row.append(dataset["aro"][i])
                row.append(dataset["val"][i])
                writer.writerow(row)

def save_anchors(filename, anchors):
    path = app.config["UPLOADS"] + filename.split(".")[0] + ".csv"
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for face in anchors["faces"]:
            row = []
            for i in range(512):
                row.append(face[0][i].item())

            writer.writerow(row)

def final_results(csv_path, anchors):
    results = []

    for i in range(len(anchors["faces"])):

        analysis = {
            "exp": [[0],[0],[0],[0],[0],[0],[0]],
            "val": [0],
            "aro": [0],
            "labels": [0]
        }

        path = csv_path + "_" + str(i) + ".csv"
        df = pd.read_csv(path) 
        data = df.values
        
        aux_secs = 0
        secs = 0
        for i, row in enumerate(data):
            if secs == aux_secs:
                for j in range(7):
                    analysis["exp"][j][-1] += row[j]

                analysis["aro"][-1] += row[7]
                analysis["val"][-1] += row[8]
            else:
                for j in range(7):
                    analysis["exp"][j].append(row[j])

                analysis["aro"].append(row[7])
                analysis["val"].append(row[8])
                analysis["labels"].append(secs)
                aux_secs += 1
            
            if ((i+1) % 2 == 0) or ((i+1) % 5 == 0):
                secs += 1
                prob_sum = 0

                for j in range(7):
                    prob_sum += analysis["exp"][j][-1]

                if prob_sum > 102:
                    for j in range(7):
                        analysis["exp"][j][-1] /= 2

                    analysis["aro"][-1] /= 2
                    analysis["val"][-1] /= 2


        for j in range(7):
            analysis["exp"][j].append(row[j])

        analysis["aro"].append(row[7])
        analysis["val"].append(row[8])
        analysis["labels"].append(secs)
        
        os.remove(path)
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(analysis["labels"])):
                row = []
                for j in range(7):
                    row.append(analysis["exp"][j][i])

                row.append(analysis["aro"][i])
                row.append(analysis["val"][i])
                writer.writerow(row)

        results.append(analysis)

    return results