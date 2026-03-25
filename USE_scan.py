import sys
import os
import cv2
import time
import numpy as np

# 같은 디렉터리의 모듈 임포트
sys.path.append(os.path.dirname(__file__))
from USE_camera_stream import StreamingTUCam
from USE_stage_test import TangoController

STREAM_WIDTH  = 1060
STREAM_HEIGHT = 800

STAGE_MAX_X=75.3169
STAGE_MAX_Y=50.1879

LENS_WIDTH=304
LENS_HEIGHT=230

SIGN_X = -1   # 픽셀 우 → 스테이지 -X
SIGN_Y = +1   # Flip=True 보정


class USE_umTomm_test:
    def __init__(self):
        self.camera = StreamingTUCam(STREAM_WIDTH, STREAM_HEIGHT)
        self.stage = TangoController()

    def pixel_to_mm_offset(self, click_x, click_y):
        """픽셀 좌표 → 스테이지 상대 이동량(mm) 변환

        Args:
            click_x: 클릭한 픽셀 X (0 ~ STREAM_WIDTH-1)
            click_y: 클릭한 픽셀 Y (0 ~ STREAM_HEIGHT-1)

        Returns:
            (dx_mm, dy_mm): 현재 위치 기준 이동해야 할 거리(mm)
        """
        dx_px = click_x - STREAM_WIDTH  / 2.0
        dy_px = click_y - STREAM_HEIGHT / 2.0

        um_per_px_x = LENS_WIDTH  / STREAM_WIDTH   # 304/1060 ≈ 0.2868 μm/px
        um_per_px_y = LENS_HEIGHT / STREAM_HEIGHT  # 230/800  = 0.2875 μm/px

        dx_mm = dx_px * um_per_px_x / 1000.0 * SIGN_X
        dy_mm = dy_px * um_per_px_y / 1000.0 * SIGN_Y

        return dx_mm, dy_mm

    def run(self, click_x, click_y):
        """클릭한 픽셀 위치로 스테이지 이동

        Args:
            click_x: 클릭한 픽셀 X
            click_y: 클릭한 픽셀 Y
        """
        dx_mm, dy_mm = self.pixel_to_mm_offset(click_x, click_y)
        print(f"[MOVE] click=({click_x}, {click_y}) → offset=({dx_mm:.6f} mm, {dy_mm:.6f} mm)")
        self.stage.move_relative(dx_mm, dy_mm, 0)


def main():
    t = USE_umTomm_test()

    # 스테이지 초기화
    if not t.stage.load_dll():
        return
    if not t.stage.create_session():
        return
    if not t.stage.connect():
        return

    pos = t.stage.get_position()
    if pos:
        print(f"[INFO] 현재 위치: X={pos[0]:.4f} Y={pos[1]:.4f} Z={pos[2]:.4f}")

    try:
        while True:
            raw = input("클릭 좌표 입력 (x y) 또는 q 종료: ").strip()
            if raw.lower() == 'q':
                break
            parts = raw.split()
            if len(parts) != 2:
                print("[ERROR] 'x y' 형식으로 입력하세요")
                continue
            try:
                cx, cy = int(parts[0]), int(parts[1])
            except ValueError:
                print("[ERROR] 정수로 입력하세요")
                continue

            if not (0 <= cx < STREAM_WIDTH and 0 <= cy < STREAM_HEIGHT):
                print(f"[ERROR] 범위 초과 (0~{STREAM_WIDTH-1}, 0~{STREAM_HEIGHT-1})")
                continue

            t.run(cx, cy)

            pos = t.stage.get_position()
            if pos:
                print(f"[INFO] 이동 후 위치: X={pos[0]:.4f} Y={pos[1]:.4f} Z={pos[2]:.4f}")

    finally:
        t.stage.disconnect()
        t.stage.free_session()


if __name__ == "__main__":
    main()
