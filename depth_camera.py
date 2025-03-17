import cv2
import numpy as np
import time
import typing
import pyrealsense2 as rs
import mediapipe as mp
import datetime as dt


class IntelRSD405:
    def __init__(
        self,
        cam_index=0,
        mtx=None,
        dist=None,
        capture_size: typing.Tuple[int, int] = (640, 480),
    ):
        super().__init__()
        self.cam_index = cam_index
        self.mtx = mtx
        self.dist = dist
        self.curr_color_frame: typing.Union[np.ndarray, None] = None
        self.capture_size = capture_size
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (20, 100)
        self.fontScale = .5
        self.color = (0,50,255)
        self.thickness = 1
        
        # ====== Realsense ======
        self.realsense_ctx = rs.context()
        self.connected_devices = [] # List of serial numbers for present cameras
        for i in range(len(self.realsense_ctx.devices)):
            detected_camera = self.realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            print(f"{detected_camera}")
            self.connected_devices.append(detected_camera)
        self.device = self.connected_devices[0] # In this example we are only using one camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.background_removed_color = 255 # Grey

        # ====== Mediapipe ======
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=0.3)
        self.mpDraw = mp.solutions.drawing_utils


    def capture(self):
        # ====== Enable Streams ======
        self.config.enable_device(self.device)

        # # For worse FPS, but better resolution:
        # stream_res_x = 1280
        # stream_res_y = 720
        # # For better FPS. but worse resolution:
        stream_res_x = 640
        stream_res_y = 480

        stream_fps = 30

        self.config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
        self.config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
        self.profile = self.pipeline.start(self.config)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # ====== Get depth Scale ======
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"\tDepth Scale for Camera SN {self.device} is: {self.depth_scale}")

        # ====== Set clipping distance ======
        self.clipping_distance_in_meters = 2
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale
        print(f"\tConfiguration Successful for SN {self.device}")

        # ====== Get and process images ====== 
        print(f"Starting to capture images on SN: {self.device}")
        

    def update_frame(self) -> bool:
        # Get and align frames
        self.frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(self.frames)
        self.aligned_depth_frame = self.aligned_frames.get_depth_frame()
        self.curr_color_frame = self.aligned_frames.get_color_frame()
        
        return True

    def color_frame(self) -> typing.Union[np.ndarray, None]:
        return self.curr_color_frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    cam = IntelRSD405(2)
    cam.capture()
    
    while True:
        start_time = dt.datetime.today().timestamp()

        # Get and align frames
        frames = cam.pipeline.wait_for_frames()
        aligned_frames = cam.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue

        # Process images
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image,1)
        color_image = np.asanyarray(color_frame.get_data())

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
        background_removed = np.where((depth_image_3d > cam.clipping_distance) | (depth_image_3d <= 0), cam.background_removed_color, color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = cv2.flip(background_removed,1)
        color_image = cv2.flip(color_image,1)
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        grayscale_3channel = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
        
        # Process hands
        # results = hands.process(color_images_rgb)
        results = cam.hands.process(grayscale_3channel)
        if results.multi_hand_landmarks:
            number_of_hands = len(results.multi_hand_landmarks)
            i=0
            for handLms in results.multi_hand_landmarks:
                cam.mpDraw.draw_landmarks(images, handLms, cam.mpHands.HAND_CONNECTIONS)
                org2 = (20, cam.org[1]+(20*(i+1)))
                hand_side_classification_list = results.multi_handedness[i]
                hand_side = hand_side_classification_list.classification[0].label
                middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]
                x = int(middle_finger_knuckle.x*len(depth_image_flipped[0]))
                y = int(middle_finger_knuckle.y*len(depth_image_flipped))
                if x >= len(depth_image_flipped[0]):
                    x = len(depth_image_flipped[0]) - 1
                if y >= len(depth_image_flipped):
                    y = len(depth_image_flipped) - 1
                mfk_distance = depth_image_flipped[y,x] * cam.depth_scale # meters
                mfk_distance_feet = mfk_distance * 3.281 # feet
                images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", 
                                     org2, cam.font, cam.fontScale, cam.color, cam.thickness, cv2.LINE_AA)
                
                i+=1
            images = cv2.putText(images, f"Hands: {number_of_hands}", 
                                 cam.org, cam.font, cam.fontScale, cam.color, cam.thickness, cv2.LINE_AA)
        else:
            images = cv2.putText(images,"No Hands", 
                                 cam.org, cam.font, cam.fontScale, cam.color, cam.thickness, cv2.LINE_AA)


        # Display FPS
        time_diff = dt.datetime.today().timestamp() - start_time
        fps = int(1 / time_diff)
        org3 = (20, cam.org[1] + 60)
        images = cv2.putText(images, f"FPS: {fps}", 
                             org3, cam.font, cam.fontScale, cam.color, cam.thickness, cv2.LINE_AA)

        name_of_window = 'SN: ' + str(cam.device)

        # Display images 
        cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name_of_window, images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {cam.device}")
            break

    print(f"Application Closing")
    cam.pipeline.stop()
    print(f"Application Closed.")
    
    
    # while True:
    #     if not cam.update_frame():
    #         continue

    #     frame = cam.color_frame()
    #     if frame is None:
    #         time.sleep(0.01)
    #         continue

    #     # print(frame.shape)
    #     window_name = "preview"
    #     cv2.imshow(window_name, frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break
