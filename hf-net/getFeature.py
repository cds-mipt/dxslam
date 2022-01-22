import cv2
import faiss
import json
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import sys
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from time import perf_counter
tf.contrib.resampler

from tqdm import tqdm

class HFNet:
    def __init__(self, model_path, outputs):
        self.session = tf.Session()
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n + ':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')

    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)

if __name__ == "__main__":

    #folders
    if len(sys.argv) < 6:
        print("Not enough arguments")
        sys.exit()
    root_dir = sys.argv[1]
    output_file = sys.argv[2]
    num_keypoints = int(sys.argv[3])
    nms_radius = int(sys.argv[4])
    path_to_database_struct = sys.argv[5] 
    output_features_path = sys.argv[6]

    os.makedirs(output_features_path, exist_ok=True)

    #define the net
    model_path = "./model/hfnet"
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
    hfnet = HFNet(model_path, outputs)

    #input the image
    # imageNames = os.listdir(imageFolder)
    # imageNames.sort()

    #create the output folder
    # localDesFolder = os.path.join(output_folder, 'des')
    # globalDesFolder = os.path.join(output_folder, 'glb')
    # keypointFolder = os.path.join(output_folder, 'point-txt')
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    # if not os.path.exists(localDesFolder):
    #     os.makedirs(localDesFolder, exist_ok=True)
    # if not os.path.exists(globalDesFolder):
    #     os.makedirs(globalDesFolder, exist_ok=True)
    # if not os.path.exists(keypointFolder):
    #     os.makedirs(keypointFolder, exist_ok=True)

    #inference
    ts = []
    with open(path_to_database_struct) as json_file:
        database_struct_dict = json.load(json_file)

    dbImage = database_struct_dict['dbImage']
    qImage = database_struct_dict['qImage']
    numDb = database_struct_dict['numDb']
    numQ = database_struct_dict['numQ']
    images = list(dbImage)
    images.extend(list(qImage))

    dbFeat = np.empty((numDb+numQ, num_keypoints))
    for num in tqdm(range(numDb+numQ)):
        filename_image = os.path.join(root_dir, images[num])
        image = cv2.imread(filename_image)
        # image = cv2.resize(image, (600, 300))
        H, W = image.shape[:2]
        t1 = perf_counter()
        hf_net_prediction = hfnet.inference(image, num_keypoints=10000, nms_radius=nms_radius)
        t2 = perf_counter()
        ts.append(t2 - t1)
        dbFeat[num] = hf_net_prediction['global_descriptor']
        
        filename_short = images[num]
        filename_short = filename_short[filename_short.rfind('/')+1:]
        filename_short = filename_short.rstrip('.png')

        map_name = images[num].split('/')[-3]
        try:
            os.makedirs(os.path.join(output_features_path, map_name), exist_ok=False)
        except:
            a = 1
        np.save(os.path.join(output_features_path, map_name, filename_short+'.npy'), hf_net_prediction['global_descriptor'])
        localDes = np.asarray(hf_net_prediction['local_descriptors'])[:num_keypoints]
        # np.save(os.path.join(localDesFolder , filename_image.split(".png")[0]), localDes)
        globalDes = np.asarray(hf_net_prediction['global_descriptor'])
        # np.save(os.path.join(globalDesFolder, filename_image.split(".png")[0]), globalDes)
        localIndex = np.asarray(hf_net_prediction['keypoints'])[:num_keypoints]
        # np.savetxt(os.path.join(keypointFolder , filename_image.split(".png")[0] + ".txt"), localIndex)
    
    qFeat = dbFeat[numDb:].astype('float32')
    dbFeat = dbFeat[:numDb].astype('float32')
    faiss_index = faiss.IndexFlatL2(num_keypoints)
    faiss_index.add(dbFeat)

    n_values = [100]

    distances, predictions = faiss_index.search(qFeat, max(n_values))

    output = {}
    results_txt_file = open(output_file[:output_file.find(".")]+'.txt', "w")
    results_txt_file.write('# HF-Net\n')
    results_txt_file.write('# query_image database_image L2_metric cos_metric\n')
    for i, prediction in enumerate(predictions):
        output[images[numDb+i]] = list([images[j] for j in prediction])
        for j, num_db in enumerate(prediction):
            dbfeature = dbFeat[num_db]
            qfeature = qFeat[i]
            dbfeature = np.reshape(dbfeature, (1,-1))
            qfeature = np.reshape(qfeature, (1,-1))
            results_txt_file.write(images[numDb+i].split('/')[-1] + \
                                   ' ' + images[num_db].split('/')[-1] + \
                                   ' ' + str(distances[i, j]) + \
                                   ' ' + str(cosine_similarity(dbfeature, qfeature)[0][0]) + '\n')
    
    with open(output_file, 'w') as fp:
        json.dump(output, fp)
    
    print('IMAGE SHAPE: {} x {}'.format(H, W))
    print('AVG INFERENCE TIME: MEAN {} STD {}'.format(np.mean(ts[1:]), np.std(ts[1:])))