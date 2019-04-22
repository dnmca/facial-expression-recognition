"""
Extracts faces from the given image.
"""

import cv2 as cv
import matplotlib.pyplot as plt


def extract_faces(image_path: str,
                  image_id: int,
                  target_dit: str,
                  image_size: int,
                  coord_path: str
                  ) -> None:
    image = cv.imread(image_path)

    image_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    haar_cascade_face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces_rects = haar_cascade_face.detectMultiScale(image_bw, scaleFactor=1.2, minNeighbors=5)

    faces = []

    # need to save faces_rects for future usage, need to save them in

    for (x, y, w, h) in faces_rects:
        y_lo = y - h // 2
        y_hi = y + h + h // 2

        x_lo = x - w // 2
        x_hi = x + w + w // 2

        faces.append(((y_lo, y_hi, x_lo, x_hi), image_bw[y_lo:y_hi, x_lo:x_hi]))

    faces = [face for face in faces if face[1].shape[0] * face[1].shape[1] != 0]

    with open(coord_path, 'w') as file:

        file.write("image_id,face_id,y_lo,y_hi,x_lo,x_hi\n")

        for index, (coords, face) in enumerate(faces):

            file.write(str(image_id) + ',' + str(index) + ',' + ','.join([str(c) for c in coords]) + '\n')

            resized = cv.resize(face, (image_size, image_size))
            plt.imsave(target_dit + str(image_id) + '_' + str(index) + '.png', resized, cmap='gray')


    return None


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract faces from image")

    parser.add_argument('--image', help='path to image', type=str)

    parser.add_argument('--dir', help='target dir to write extracted faces', type=str)

    parser.add_argument('--coord', help='file to write bounding boxes coordinates', type=str)

    parser.add_argument('--id', help='image id', type=int)

    parser.add_argument('--size', help='size of the output image', type=int, default=48)

    arguments = parser.parse_args()

    extract_faces(arguments.image, arguments.id, arguments.dir, arguments.size, arguments.coord)
