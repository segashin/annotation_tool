import os
import sys
import cv2
import xml.etree.ElementTree as ET


def exportXML(coords, label, img_dims, img_name, img_path, out_path):

    # create the file structure
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    filename = ET.SubElement(annotation, 'filename')
    path = ET.SubElement(annotation, 'path')
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    segmented = ET.SubElement(annotation, 'segmented')
    object = ET.SubElement(annotation, 'object')
    name = ET.SubElement(object, 'name')
    pose = ET.SubElement(object, 'pose')
    truncated = ET.SubElement(object, 'truncated')
    difficult = ET.SubElement(object, 'difficult')
    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    ymin = ET.SubElement(bndbox, 'ymin')
    xmax = ET.SubElement(bndbox, 'xmax')
    ymax = ET.SubElement(bndbox, 'ymax')

    annotation.set('verified', 'yes')

    folder.text = "images"
    filename.text = img_name
    path.text = img_path
    database.text = "CS497 Sign Language"
    width.text = str(img_dims[0])
    height.text = str(img_dims[1])
    depth.text = str(img_dims[2])
    segmented.text = "0"
    name.text = label
    pose.text = "Unspecified"
    truncated.text = "0"
    difficult.text = "0"
    xmin.text = str(coords[0][0])
    ymin.text = str(coords[0][1])
    xmax.text = str(coords[1][0])
    ymax.text = str(coords[1][1])

    # create a new XML file with the results
    img_annot = ET.tostring(annotation)
    with open(out_path, 'wb') as f:
        f.write(img_annot)


def define_rect(image):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    rect_pts = []  # Starting and ending points
    win_name = "image"  # Window name

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"):  # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord(" "):  # Hit 'c' to confirm the selection
            break
        elif key == ord("q"):
            return None

    # close the open windows
    cv2.destroyWindow(win_name)

    return rect_pts


def main(annotator_id, video_file_name, label, num_frames_to_export):
    cap = cv2.VideoCapture(video_file_name)

    annot_dir_name = "./annotations"
    img_dir_name = "./images"
    os.listdir(annot_dir_name)
    count = 0
    while True:
        if count == num_frames_to_export:
            exit()

        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        coords = define_rect(frame)

        if coords is None:
            continue

        numImgs = len([f for f in os.listdir(img_dir_name) if f.endswith('.png')
                       and os.path.isfile(os.path.join(img_dir_name, f))])
        id = "{0:04d}".format(numImgs + 1)

        img_dims = frame.shape
        img_name = f"img_{annotator_id}_{id}.png"

        img_path = os.path.join(img_dir_name, img_name)
        out_path = os.path.join(
            annot_dir_name, f"annot_{annotator_id}_{id}.xml")

        count += 1
        exportXML(coords, label, img_dims,
                  img_name, img_path, out_path)

        cv2.imwrite(img_path, frame)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python video2annotatedFrames.py <annotator_id> <video_file_name> <label> <num_frames_to_export>")
        exit()

    # annotator_id aims to prevent file name collision when making data separately
    annotator_id = sys.argv[1]
    video_file_name = sys.argv[2]
    label = sys.argv[3]
    num_frames_to_export = int(sys.argv[4])

    main(annotator_id, video_file_name, label, num_frames_to_export)
