import os
import shutil
from datetime import datetime
from moviepy.editor import VideoFileClip
import cv2
import torch

from src.ucs_alg_node import Alg, Fcn

video_resize = 0.25

model = Fcn()  # 定义模型
weights_path = "test/weights/model_12.pth"  # 权重
weights = torch.load(weights_path, map_location='cpu')#cpu
model.load_state_dict(weights)


class MyAlg(Alg):
    def infer_batch(self, data):
        if not data:
            return []

        if not os.path.isdir(data):
            print("The provided data path is not a directory.")
            return []

        files = os.listdir(data)
        print(f"Files in directory: {files}")
        videos = [file for file in files if file.endswith('.mp4')]  # Filter for .mp4 files

        if not videos:
            print("No video files found in the directory.")
            return []


        # return videos
        # 存储每个视频帧数
        frame_count_list = []

        for video in videos:
            # 读取路径
            video_path = os.path.join(data, video)
            cap = cv2.VideoCapture(video_path)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_count_list.append(frame_count)

            file_name, file_extension = os.path.splitext(video)
            new_file_name = f"{file_name}_{frame_count}{file_extension}"
            # 存放新视频的文件夹路径
            output_name = os.path.join(data,'frame_video')
            folder = os.path.exists(output_name)

            if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(output_name)

            else:
                pass
            output_path = os.path.join(output_name, new_file_name)

            # 获取视频的宽度、高度和帧率
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 定义视频编解码器并创建 VideoWriter 对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
            out = cv2.VideoWriter(output_path, fourcc, frame_count,(frame_width, frame_height))
            cap.release()
            out.release()
        # TODO: 从data所指定的地址读取视频，并且存储到本地: ‘video___ts.mp4’
        f = open('video.mp4', 'wb')
        cap = cv2.VideoCapture(f)  # 读取视频
        ret, img = cap.read()  # 取出
        if ret == False:
            print("Failed to read video")
            cap.release()
            cv2.destroyAllWindows()
            return []

        img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)  # 输入模型的第一帧图片，缩放
        img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)  # 二值
        prev_img_grey = img_grey  # 第一帧视频
        frame = 1  # 第二帧视频

        labels = []

        while True:
            ret, img = cap.read()  # 取出视频， img是原始视频

            if not ret:
                # print("读取视频结束")
                break

            img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)  # 输入视频
            img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_img_grey, img_grey, None, 0.5, 5, 15, 3, 5, 1.1,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            flow = flow[np.newaxis, :]
            flow = flow.transpose(0, 3, 1, 2)
            flow = torch.from_numpy(flow).float().to(model.device)

            predict = model(flow)  # 预测输出
            predict = torch.argmax(predict, 1).cpu().numpy()[0]

            labels.append(predict)

            frame += 1  #
            prev_img_grey = img_grey

        cap.release()
        cv2.destroyAllWindows()

        # TODO: 删除临时文件
        os.remove('video.mp4')
        return labels
if __name__ == "__main__":
    data = r'D:\gugol\Motion_Emotion_Dataset'
    alg = MyAlg()
    videos = alg.infer_batch(data)
    frame_count_list =[]
    frame_list =[]
    for video in videos:
        video_path =os.path.join(data, video)
        capture = cv2.VideoCapture(video_path)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count_list.append(frame_count)


        # cap = cv2.VideoCapture(video_path)
        # frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 通过属性获取帧数
        # count = 0  # 用于计算视频的实际帧数
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     count += 1
        # frame_list.append(count)
    print(frame_count_list)
    print(frame_list)
    # dst_paths = alg.copy_videos_and_return_paths(videos, data)

    # print(dst_paths)


    # for video in videos:
    #     video_path = os.path.join(r'D:\\gugol\\Motion_Emotion_Dataset', video)
    #     labels = alg.process_video(dst_path)
    #     print(f"Processed {video}: {labels}")