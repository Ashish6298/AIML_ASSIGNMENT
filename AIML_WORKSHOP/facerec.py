






# import cv2 as cv
# import argparse
# import numpy as np
# import os

# def validate_image_path(path):
#     if not os.path.isfile(path):
#         raise FileNotFoundError(f"The file {path} does not exist.")
#     return path

# def validate_directory_path(path):
#     if not os.path.isdir(path):
#         raise FileNotFoundError(f"The directory {path} does not exist.")
#     return path

# def visualize(image, faces, thickness=2):
#     if faces[1] is not None:
#         for idx, face in enumerate(faces[1]):
#             coords = face[:-1].astype(np.int32)
#             cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
#             colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
#             for i in range(5):
#                 cv.circle(image, (coords[4 + 2*i], coords[5 + 2*i]), 2, colors[i], thickness)
#             cv.putText(image, f'Face {idx+1}', (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-r", "--reference_images_dir", required=True, help="Path to directory containing Aadhaar reference images", type=validate_directory_path)
#     ap.add_argument("-q", "--query_image", required=True, help="Path to input query image", type=validate_image_path)
#     args = vars(ap.parse_args())

#     reference_images_dir = args["reference_images_dir"]
#     query_image_path = args["query_image"]

#     print(f"Reference images directory: {reference_images_dir}")
#     print(f"Query image path: {query_image_path}")

#     query_image = cv.imread(query_image_path)
#     if query_image is None:
#         print(f"Failed to load query image: {query_image_path}")
#         return

#     score_threshold = 0.9
#     nms_threshold = 0.3
#     top_k = 5000

#     faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "",
#                                             (query_image.shape[1], query_image.shape[0]), 
#                                             score_threshold, nms_threshold, top_k)

#     faceDetector.setInputSize((query_image.shape[1], query_image.shape[0]))
#     faceInQuery = faceDetector.detect(query_image)
#     visualize(query_image, faceInQuery)
#     cv.imshow("Face in Query", query_image)
#     cv.waitKey(0)

#     recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")
#     cosine_similarity_threshold = 0.363
#     l2_similarity_threshold = 1.128

#     total_images = 0
#     detections = 0
#     matches_cosine = 0
#     matches_l2 = 0

#     matched_aadhaar_images = []

#     for filename in os.listdir(reference_images_dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             total_images += 1
#             reference_image_path = os.path.join(reference_images_dir, filename)
#             print(f"Processing reference image: {reference_image_path}")

#             reference_image = cv.imread(reference_image_path)
#             if reference_image is None:
#                 print(f"Failed to load reference image: {reference_image_path}")
#                 continue

#             faceDetector.setInputSize((reference_image.shape[1], reference_image.shape[0]))
#             faceInAadhaar = faceDetector.detect(reference_image)
#             visualize(reference_image, faceInAadhaar)
#             cv.imshow(f"Face in {filename}", reference_image)
#             cv.waitKey(0)

#             if faceInAadhaar[1] is not None and faceInQuery[1] is not None:
#                 detections += 1
#                 face1_align = recognizer.alignCrop(reference_image, faceInAadhaar[1][0])
#                 face2_align = recognizer.alignCrop(query_image, faceInQuery[1][0])

#                 if face1_align is not None and face2_align is not None:
#                     face1_feature = recognizer.feature(face1_align)
#                     face2_feature = recognizer.feature(face2_align)

#                     cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
#                     l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)

#                     if cosine_score >= cosine_similarity_threshold:
#                         matches_cosine += 1
#                         matched_aadhaar_images.append(filename)
#                     if l2_score <= l2_similarity_threshold:
#                         matches_l2 += 1

#     detection_accuracy = (detections / total_images) * 100
#     match_accuracy_cosine = (matches_cosine / detections) * 100 if detections > 0 else 0
#     match_accuracy_l2 = (matches_l2 / detections) * 100 if detections > 0 else 0

#     print(f'Detection Accuracy: {detection_accuracy:.2f}%')
#     print(f'Matching Accuracy (Cosine Similarity): {match_accuracy_cosine:.2f}%')
#     print(f'Matching Accuracy (L2 Similarity): {match_accuracy_l2:.2f}%')

#     if matched_aadhaar_images:
#         print("Matched Aadhaar Card Images:")
#         for matched_image in matched_aadhaar_images:
#             print(matched_image)

# if __name__ == "__main__":
#     main()







import cv2 as cv
import argparse
import numpy as np
import os

def validate_image_path(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    return path

def validate_directory_path(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")
    return path

def visualize(image, faces, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
            colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
            for i in range(5):
                cv.circle(image, (coords[4 + 2*i], coords[5 + 2*i]), 2, colors[i], thickness)
            cv.putText(image, f'Face {idx+1}', (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--reference_images_dir", required=True, help="Path to directory containing Aadhaar reference images", type=validate_directory_path)
    ap.add_argument("-q", "--query_image", required=True, help="Path to input query image", type=validate_image_path)
    args = vars(ap.parse_args())

    reference_images_dir = args["reference_images_dir"]
    query_image_path = args["query_image"]

    print(f"Reference images directory: {reference_images_dir}")
    print(f"Query image path: {query_image_path}")

    query_image = cv.imread(query_image_path)
    if query_image is None:
        print(f"Failed to load query image: {query_image_path}")
        return

    # Adjust detection parameters
    score_threshold = 0.8
    nms_threshold = 0.2
    top_k = 10000

    faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "",
                                            (query_image.shape[1], query_image.shape[0]), 
                                            score_threshold, nms_threshold, top_k)

    faceDetector.setInputSize((query_image.shape[1], query_image.shape[0]))
    faceInQuery = faceDetector.detect(query_image)
    visualize(query_image, faceInQuery)
    cv.imshow("Face in Query", query_image)
    cv.waitKey(0)

    recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")
    cosine_similarity_threshold = 0.4
    l2_similarity_threshold = 1.0

    total_images = 0
    detections = 0
    matches_cosine = 0
    matches_l2 = 0

    matched_aadhaar_images = []

    for filename in os.listdir(reference_images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            total_images += 1
            reference_image_path = os.path.join(reference_images_dir, filename)
            print(f"Processing reference image: {reference_image_path}")

            reference_image = cv.imread(reference_image_path)
            if reference_image is None:
                print(f"Failed to load reference image: {reference_image_path}")
                continue

            faceDetector.setInputSize((reference_image.shape[1], reference_image.shape[0]))
            faceInAadhaar = faceDetector.detect(reference_image)
            visualize(reference_image, faceInAadhaar)
            cv.imshow(f"Face in {filename}", reference_image)
            cv.waitKey(0)

            if faceInAadhaar[1] is not None and faceInQuery[1] is not None:
                detections += 1
                face1_align = recognizer.alignCrop(reference_image, faceInAadhaar[1][0])
                face2_align = recognizer.alignCrop(query_image, faceInQuery[1][0])

                if face1_align is not None and face2_align is not None:
                    face1_feature = recognizer.feature(face1_align)
                    face2_feature = recognizer.feature(face2_align)

                    cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
                    l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)

                    if cosine_score >= cosine_similarity_threshold:
                        matches_cosine += 1
                        matched_aadhaar_images.append(filename)
                    if l2_score <= l2_similarity_threshold:
                        matches_l2 += 1

    detection_accuracy = (detections / total_images) * 100
    match_accuracy_cosine = (matches_cosine / detections) * 100 if detections > 0 else 0
    match_accuracy_l2 = (matches_l2 / detections) * 100 if detections > 0 else 0

    print(f'Detection Accuracy: {detection_accuracy:.2f}%')
    print(f'Matching Accuracy (Cosine Similarity): {match_accuracy_cosine:.2f}%')
    print(f'Matching Accuracy (L2 Similarity): {match_accuracy_l2:.2f}%')

    if matched_aadhaar_images:
        print("Matched Aadhaar Card Images:")
        for matched_image in matched_aadhaar_images:
            print(matched_image)

if __name__ == "__main__":
    main()

#  python facerec.py -r "reference_images_dir/" -q "query_image.jpg" (to run)




