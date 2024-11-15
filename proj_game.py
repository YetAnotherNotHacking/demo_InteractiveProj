import cv2
import mediapipe as mp
import numpy as np
import pygame
from cv2 import aruco
import math

class InteractiveProjectorTracker:
    def __init__(self, camera_index=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Failed to open camera {camera_index}, trying camera 0...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise ValueError("No camera found!")
        self.calibration_points = []
        self.calibration_complete = False
        self.transform_matrix = None
        self.projector_width = 1280
        self.projector_height = 840
        self.projector_window = np.zeros((self.projector_height, self.projector_width, 3), np.uint8)
        self.projector_window.fill(255)
        self.create_projector_window()
        pygame.init()
        self.screen = pygame.display.set_mode((self.projector_width, self.projector_height))
        self.ball_pos = [self.projector_width // 2, self.projector_height // 2]
        self.ball_vel = [0, 0]
        self.ball_radius = 20
        self.gravity = 0.5
        self.friction = 0.9

    def create_projector_window(self):
        cv2.namedWindow("Projector View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Projector View", self.projector_width, self.projector_height)

    def calibrate(self):
        calibration_points_screen = [(100, 100), (self.projector_width - 100, 100), (100, self.projector_height - 100), (self.projector_width - 100, self.projector_height - 100)]
        print("Starting calibration... Point to each calibration point as they appear and press SPACE")
        self.calibration_points = []
        for i, point in enumerate(calibration_points_screen):
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                self.projector_window.fill(0)
                cv2.circle(self.projector_window, point, 20, (0, 255, 0), -1)
                cv2.imshow("Projector View", self.projector_window)
                finger_pos = self.detect_finger(frame)
                if finger_pos:
                    cv2.circle(frame, finger_pos, 5, (0, 0, 255), -1)
                cv2.imshow('Camera View', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and finger_pos:
                    self.calibration_points.append(finger_pos)
                    break
                elif key == ord('q'):
                    return False
        if len(self.calibration_points) == 4:
            self.transform_matrix = cv2.getPerspectiveTransform(np.float32(self.calibration_points), np.float32(calibration_points_screen))
            self.calibration_complete = True
            print("Calibration complete!")
            return True
        return False

    def detect_finger(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                frame_height, frame_width = frame.shape[:2]
                x = int(index_tip.x * frame_width)
                y = int(index_tip.y * frame_height)
                return (x, y)
        pose_results = self.pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            wrist = pose_results.pose_landmarks.landmark[15]
            frame_height, frame_width = frame.shape[:2]
            x = int(wrist.x * frame_width)
            y = int(wrist.y * frame_height)
            return (x, y)
        return None

    def map_to_projector(self, finger_pos):
        if not self.calibration_complete or finger_pos is None:
            return None
        point = np.array([finger_pos], dtype=np.float32)
        point = np.array([point])
        transformed_point = cv2.perspectiveTransform(point, self.transform_matrix)
        return tuple(map(int, transformed_point[0][0]))

    def run(self):
        if not self.calibrate():
            print("Calibration failed or was cancelled")
            return
        running = True
        while running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.projector_window.fill(0)
            self.screen.fill((0, 0, 0))
            finger_pos = self.detect_finger(frame)
            if finger_pos:
                cv2.circle(frame, finger_pos, 5, (0, 0, 255), -1)
                projector_pos = self.map_to_projector(finger_pos)
                if projector_pos:
                    self.handle_ball_interaction(projector_pos)
                    pygame.draw.circle(self.screen, (255, 0, 0), self.ball_pos, self.ball_radius)
            elif not self.calibration_complete:
                cv2.putText(self.projector_window, "Calibration missing, press 'c' to calibrate", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Camera View', frame)
            projector_surface = pygame.surfarray.array3d(self.screen)
            self.projector_window = cv2.cvtColor(projector_surface, cv2.COLOR_RGB2BGR)
            cv2.imshow('Projector View', self.projector_window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
            elif cv2.waitKey(1) & 0xFF == ord('c'):
                self.calibrate()
            self.apply_ball_physics()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

    def apply_ball_physics(self):
        self.ball_vel[1] += self.gravity
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        if self.ball_pos[1] >= self.projector_height - self.ball_radius:
            self.ball_pos[1] = self.projector_height - self.ball_radius
            self.ball_vel[1] *= -self.friction
        if self.ball_pos[0] <= self.ball_radius or self.ball_pos[0] >= self.projector_width - self.ball_radius:
            self.ball_vel[0] *= -self.friction

    def handle_ball_interaction(self, hand_pos):
        hand_x, hand_y = hand_pos
        dist = math.sqrt((self.ball_pos[0] - hand_x) ** 2 + (self.ball_pos[1] - hand_y) ** 2)
        if dist < self.ball_radius:
            dx = self.ball_pos[0] - hand_x
            dy = self.ball_pos[1] - hand_y
            self.ball_vel[0] += dx * 0.1
            self.ball_vel[1] += dy * 0.1

if __name__ == "__main__":
    tracker = InteractiveProjectorTracker(camera_index=2)
    tracker.run()
