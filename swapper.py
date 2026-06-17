import cv2
import numpy as np
import pybase64 
from openvino import Core, Tensor
import os
import sys
import io
import gc
import time
import random
import string
from typing import List,Tuple,Union, Optional, Dict, Any
np.set_printoptions(suppress=True)
from utils.common import distance2bbox,distance2kps,generate_anchor_centers_batch
from utils.aligner import FaceAligner
from utils.smallFunctions import random_alphanumeric,is_image_empty,im2b64,get_sim,binary_quantize,hamming_distance,clear_vectors,load_image


def clear_vectors(vectors_variable):
    """Clears a loaded vectors variable and releases memory."""

    if vectors_variable is not None:
        try:
            del vectors_variable # Delete the variable reference
            gc.collect() # Force garbage collection
#            print("Vectors variable cleared.")
        except NameError:
            print("Variable does not exist, or has already been cleared.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Vectors variable was already None.")

    
class sdeeRecognition:
    def __init__(self, model: str = "buffalo_l"):
#    def __init__(self, model: str = "buffalo_sc"):
        self.model_name = model  # Store model name
        self._init_vars()

    def _init_vars(self) -> None:
        self.center_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.ie: Core = Core()
        self.config = {
            "PERFORMANCE_HINT": "THROUGHPUT"
        }

        self.config1 = {
            "PERFORMANCE_HINT": "LATENCY"
        }
        
        self.aligner: FaceAligner = FaceAligner() 
        self.compiled_model: openvino._ov_api.CompiledModel
        self.embedding_compiled_model: openvino._ov_api.CompiledModel
        self.swap_compiled_model: openvino._ov_api.CompiledModel

        self.compiled_model, self.embedding_compiled_model,self.swap_compiled_model = self.loadModels()
        self.swap_request = self.swap_compiled_model.create_infer_request()
        self.embedding_request = self.embedding_compiled_model.create_infer_request()

        print("Warming Up ")
        for _ in range(3):
            self.embedding_request.infer()
            self.swap_request.infer()
        print("Done")
        
        self.input_layer: ConstOutput = self.compiled_model.input(0) # Replace Any with the actual type from OpenVINO
        self.detect_input_size = self.input_layer.shape[2]
        self.reco_input_size: int = self.embedding_compiled_model.inputs[0].shape[2]
        self.swap_input_size: Tuple = (128,128)
        print("det size", self.detect_input_size, "Reco Size", self.reco_input_size, "Swap Size", self.swap_input_size)
        self.output_layer: List[list] = self.compiled_model.outputs # Replace Any with the actual type from OpenVINO
        print('Loading EMAP')
        self.emap = np.load("emap.npy")
        print('EMAP', self.emap.shape)

        self.input_mean = 0.0
        self.input_std = 255.0

    def loadModels(self) -> Tuple[Any, Any]:
        self.modelToLoad = self.model_name.split("_")[0] 
#        embedding_model: str = f"./weights/{self.model_name}.blob"
#        swap_model: str = f"./weights/compiled_inswapper_128.blob"
#        swap_model: str = f"./weights/compiled_inswapper_128_FP16.blob"
     
        start: float = time.time()
        model:  openvino._pyopenvino.Model = self.ie.read_model(model="./weights/det_10g.xml") # Replace Any with the actual type from OpenVINO
        compiled_model: openvino._ov_api.CompiledModel = self.ie.compile_model(model, 'CPU', self.config1)
        end: float = time.time()
        print(f"result Loading Detection model det_10g.xml: time: {end - start}")

        gc.collect()
        time.sleep(3)
        start = time.time()
        swap_model: openvino._pyopenvino.Model = self.ie.read_model(model="./weights/inswapper_128.xml")
        swap_compiled_model: openvino._ov_api.CompiledModel = self.ie.compile_model(swap_model,'CPU', self.config)          
        # with open(swap_model, "rb") as f:
        #     model_stream: bytes = f.read()
        # swap_compiled_model: openvino._ov_api.CompiledModel = self.ie.import_model(model_stream,'CPU')
        end = time.time()
        print(f"result Loading Swapping model inswapper_128: time: {end - start}")

        gc.collect()
        time.sleep(3)
        start = time.time()
        embedding_model: openvino._pyopenvino.Model = self.ie.read_model(model="./weights/buffalo_l.xml")
        embedding_compiled_model: openvino._ov_api.CompiledModel = self.ie.compile_model(embedding_model, 'CPU', self.config1)         
        
        # with open(embedding_model, "rb") as f:
        #     model_stream: bytes = f.read()
        # embedding_compiled_model: openvino._ov_api.CompiledModel = self.ie.import_model(model_stream,'CPU')
        end = time.time()
        print(f"result Loading Recognition model {self.model_name}: time: {end - start}")

        return compiled_model, embedding_compiled_model, swap_compiled_model

    def process_network_outputs_batch(self, net_outs: List[np.ndarray], threshold: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []
        fmc: int = 3
        strides: np.ndarray = np.array([8, 16, 32])
        heights: np.ndarray = self.detect_input_size // strides
        widths: np.ndarray = self.detect_input_size // strides
        all_anchor_centers: List[np.ndarray] = generate_anchor_centers_batch(heights, widths, strides, 2, self.center_cache)

        for idx, stride in enumerate(strides):
            scores: np.ndarray = net_outs[idx]
            bbox_preds: np.ndarray = net_outs[idx + fmc]
            kps_preds: np.ndarray = net_outs[idx + fmc * 2]
            anchor_centers: np.ndarray = all_anchor_centers[idx]

            bbox_preds = bbox_preds * stride
            bboxes: np.ndarray = distance2bbox(anchor_centers, bbox_preds)
            pos_inds: np.ndarray = np.where(scores >= threshold)[0]
            pos_scores: np.ndarray = scores[pos_inds]
            pos_bboxes: np.ndarray = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss: np.ndarray = distance2kps(anchor_centers, kps_preds * stride)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss: np.ndarray = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def forward(self, img: np.ndarray, threshold: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        blob: np.ndarray = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.detect_input_size, self.detect_input_size), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.compiled_model({self.input_layer.any_name: blob})
        net_outs: List[np.ndarray] = [outputs[name] for name in self.output_layer]
        fmc: int = 3
        return self.process_network_outputs_batch(net_outs, threshold)

    def detect_faces(self, img: np.ndarray, max_num: int = 0, metric: str = 'default', threshold: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if is_image_empty(img):
#            print('Image is empty check the path')
            return None, None

        im_ratio: float = float(img.shape[0]) / img.shape[1]
        model_ratio: float = 1.0
        if im_ratio > model_ratio:
            new_height: int = self.detect_input_size
            new_width: int = int(new_height / im_ratio)
        else:
            new_width: int = self.detect_input_size
            new_height: int = int(new_width * im_ratio)
        det_scale: float = float(new_height) / img.shape[0]
        resized_img: np.ndarray = cv2.resize(img, (new_width, new_height))
        det_img: np.ndarray = np.zeros((self.detect_input_size, self.detect_input_size, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, threshold)

        scores: np.ndarray = np.vstack(scores_list)
        scores_ravel: np.ndarray = scores.ravel()
        order: np.ndarray = scores_ravel.argsort()[::-1]
        bboxes: np.ndarray = np.vstack(bboxes_list) / det_scale
        kpss: np.ndarray = np.vstack(kpss_list) / det_scale
        pre_det: np.ndarray = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep: List[int] = self.nms(pre_det)
        det: np.ndarray = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if max_num > 0 and det.shape[0] > max_num:
            area: np.ndarray = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center: Tuple[int, int] = img.shape[0] // 2, img.shape[1] // 2
            offsets: np.ndarray = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared: np.ndarray = np.sum(np.power(offsets, 2.0), 0)
            values: np.ndarray = area if metric == 'max' else area - offset_dist_squared * 2.0
            bindex: np.ndarray = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]

            if kpss is not None:
                kpss = kpss[bindex, :]
        if len(det) >= 1:
            return det[0], kpss
        return None,None

    def alignAndCrop(self,img: np.ndarray,kpss: np.ndarray) -> np.ndarray:
        return self.aligner.norm_crop(img,kpss,self.reco_input_size)

    def get_embedding(self, cropped_face: np.ndarray) -> np.ndarray:
      print("model:", self.modelToLoad)
      if self.modelToLoad == "buffalo":
          blob = cv2.dnn.blobFromImages([cropped_face],scalefactor=1.0 / 127.5,size=(self.reco_input_size, self.reco_input_size), mean=(127.5, 127.5, 127.5), swapRB=True)

      tensor = Tensor(blob)
      infer_request = self.embedding_compiled_model.create_infer_request()
      infer_request.set_input_tensor(tensor)
      infer_request.infer()
      print("embedding Done")      
      return infer_request.get_output_tensor(0).data[0] 


    def nms(self, dets: np.ndarray) -> np.ndarray:
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
        
    def get_sim(self,emb1: np.ndarray, emb2: np.ndarray) -> float:
         dot_product = np.dot(emb1,emb2)
         norm_query = np.linalg.norm(emb2)
         norms = np.linalg.norm(emb1, axis=0)
         similarity = dot_product / (norm_query * norms + 1e-8)  # Avoid division by zero
         return similarity


    def paste_back_simple(self,target_img,swapped_face,M):
        IM = cv2.invertAffineTransform(M)
        warped_face = cv2.warpAffine(
            swapped_face,
            IM,
            (target_img.shape[1], target_img.shape[0]),
            flags=cv2.INTER_LINEAR
        )


        mask = np.zeros((128,128), dtype=np.uint8)

        cv2.rectangle(
            mask,
            (4,4),
            (123,123),
            255,
            -1
        )

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=3)

        mask = cv2.GaussianBlur(mask, (31,31), 0)

        warped_mask = cv2.warpAffine(
            mask,
            IM,
            (target_img.shape[1], target_img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
        warped_mask = (
            warped_mask.astype(np.float32)
            / 255.0
        )
        warped_mask = warped_mask[...,None]

        result = (
            warped_face.astype(np.float32) * warped_mask +
            target_img.astype(np.float32) * (1.0 - warped_mask)
        )

        return result.astype(np.uint8)
        

    def verify (self,im1: Union[str, bytes],im2: Union[str, bytes]) -> bool :
        im1 = load_image(im1)          
        im2 = load_image(im2)
        if im1 is None or im2 is None:
          return {"status":"error","message":"Path of one of the files not found"}
        dets: Optional[np.ndarray]
        kpss: Optional[np.ndarray]
        dets1, kpss1 = self.detect_faces(im1)
        dets2, kpss2 = self.detect_faces(im2)
        if dets1 is None or dets2 is None:
          return {"status":"error","message":"No face detected in one of the files"}
#        print(dets1)
        self.crop_f1, M = self.alignAndCrop(im1,kpss1[0])
        self.crop_f2. M = self.alignAndCrop(im2,kpss2[0])
        self.vector1 = self.get_embedding(self.crop_f1)
        self.vector2 = self.get_embedding(self.crop_f2)
        similarity = self.get_sim(self.vector1,self.vector2)
        clear_vectors(self.vector1)
        clear_vectors(self.vector2)
        resp ="Nomatch" if similarity < 0.5 else "Match"
        return {"status":resp, "similarity":similarity}




    def swap (self,im1: Union[str, bytes],im2: Union[str, bytes], paste_back: bool = True) -> bool :
        im1 = load_image(im1)          
        im2 = load_image(im2)

        if im1 is None or im2 is None:
          return {"status":"error","message":"Path of one of the files not found"}
        dets1: Optional[np.ndarray]
        dets2: Optional[np.ndarray]
        kpss1: Optional[np.ndarray]
        kpss2: Optional[np.ndarray]



        print("")
        print("===========================")

        t0 = time.perf_counter()    
        dets1, kpss1 = self.detect_faces(im1)
        print("detect:", time.perf_counter()-t0)
        
        t0 = time.perf_counter()
        dets2, kpss2 = self.detect_faces(im2)
        print("detect:", time.perf_counter()-t0)
        
        # dets1, kpss1 = self.detect_faces(im1)
        # dets2, kpss2 = self.detect_faces(im2)
        if dets1 is None or dets2 is None:
          return {"status":"error","message":"No face detected in one of the files"}
        self.crop_f1, M = self.alignAndCrop(im1,kpss1[0])

        t0 = time.perf_counter()
        self.vector1 = self.get_embedding(self.crop_f1).astype(np.float32)
        print("embedding:", time.perf_counter()-t0)

        
        norm1 = np.linalg.norm(self.vector1)
        if norm1 > 0: self.vector1 /= norm1
        
        latent = self.vector1.reshape(1, -1)
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        swap_cropped, M = self.aligner.norm_crop(im2,kpss2[0],128)

        blob = cv2.dnn.blobFromImage(swap_cropped, 1.0 / self.input_std, self.swap_input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        self.swap_request.set_tensor("target", Tensor(blob))
        self.swap_request.set_tensor("source", Tensor(latent.astype(np.float32)))


        t0 = time.perf_counter()
        self.swap_request.infer()
        print("swap:", time.perf_counter()-t0)
        
        pred = self.swap_request.get_output_tensor(0).data

        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        clear_vectors(self.vector1)

        if not paste_back:
            return bgr_fake
        else:
            target_img = im2
            t0 = time.perf_counter()
            fake_merged = self.paste_back_simple(target_img,bgr_fake,M)
            print("blend:", time.perf_counter()-t0)
            return fake_merged

          
    def process_image(self, image_input: Union[str, bytes], save_cropped: bool = False, returnBinaryVector: bool = False) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray]]:
        img: Optional[np.ndarray] = load_image(image_input)
        if img is None:
          return None,None
        results: List[Dict[str, Any]] = []
        if is_image_empty(img):
            face_data = {"status":"error","message": "Image is Empty"}
            cropped_face=None
        dets: Optional[np.ndarray]
        kpss: Optional[np.ndarray]
        dets, kpss = self.detect_faces(img)
        if dets is None:
            face_data = {"status":"error","message": "No Face Detected"}
            cropped_face = None
        else:
            face_data: Dict[str, Any] = {
                "bbox": dets[0:4].astype(np.int32),
                "landmarks": kpss.astype(np.int32),
                "score": float(dets[4])
            }
            cropped_face: np.ndarray = self.alignAndCrop(img, kpss[0])
            if save_cropped:
                os.makedirs("./croppedFaces/", exist_ok=True)
                cropped_path: str = f"cropped_face_{random_alphanumeric()}.jpg"
#                cv2.imwrite("./croppedFaces/" + cropped_path, cropped_face)
                _, img_encoded = cv2.imencode('.jpg', cropped_face)
                with open("./croppedFaces/" + cropped_path, "wb") as f:
                  f.write(img_encoded.tobytes())
                face_data["cropped_image"] = "./croppedFaces/" + cropped_path
            face_data["status"] = "success"
            face_data["message"] = "Face detected,aligned,croped and Embedding created in => results[0]"
            emb = self.get_embedding(cropped_face)
            if returnBinaryVector:
              emb = binary_quantize(emb, threshold=0.2)
            face_data["embedding"] = emb
        results.append(face_data)
        return results, cropped_face

#faceUtils = sdeeRecognition()

models =["buffalo_l","buffalosc","facenet","vgg"]

detector = sdeeRecognition(model=models[0])

im1 = "./images/ana.jpg"
im2 = "./images/p.jpg"

i = 0

while i <= 2:

    res = detector.swap(im1, im2, True)
    end = time.time()
    cv2.imwrite("./Vino_swapped.jpg", res)
    i += 1
    time.sleep(1)
    
#print(sim)
exit()
