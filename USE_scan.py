import sys
import os
import cv2
import time
import numpy as np

# 같은 디렉터리의 모듈 임포트
sys.path.append(os.path.dirname(__file__))
from USE_camera_stream import StreamingTUCam
from USE_stage_test import TangoController
from USE_sam3 import SAM3Segmenter

STREAM_WIDTH  = 1060
STREAM_HEIGHT = 800

STAGE_MAX_X=75.3169
STAGE_MAX_Y=50.1879

LENS_WIDTH=304
LENS_HEIGHT=230

SIGN_X = -1   # 픽셀 우 → 스테이지 -X
SIGN_Y = +1   # Flip=True 보정


class USE_scan:
    def __init__(self):
        self.camera = StreamingTUCam(STREAM_WIDTH, STREAM_HEIGHT)
        self.stage = TangoController()
        self.sam3 = SAM3Segmenter()

    def pixel_to_mm_offset(self, rel_x, rel_y):
        """픽셀 좌표 → 스테이지 상대 이동량(mm) 변환

        Args:
            click_x: 클릭한 픽셀 X (0 ~ STREAM_WIDTH-1)
            click_y: 클릭한 픽셀 Y (0 ~ STREAM_HEIGHT-1)

        Returns:
            (dx_mm, dy_mm): 현재 위치 기준 이동해야 할 거리(mm)
        """
        dx_px = rel_x - STREAM_WIDTH  / 2.0
        dy_px = rel_y - STREAM_HEIGHT / 2.0

        um_per_px_x = LENS_WIDTH  / STREAM_WIDTH   # 304/1060 ≈ 0.2868 μm/px
        um_per_px_y = LENS_HEIGHT / STREAM_HEIGHT  # 230/800  = 0.2875 μm/px

        dx_mm = dx_px * um_per_px_x / 1000.0 * SIGN_X
        dy_mm = dy_px * um_per_px_y / 1000.0 * SIGN_Y

        return dx_mm, dy_mm

    def sam_to_absolute(self, detected_objects: list) -> list:
        """SAM JSON 좌표를 스테이지 절대좌표(mm)로 변환.

        단위 흐름: px → × (μm/px) → μm → ÷ 1000 → mm
        TangoController(get_position / move_absolute)는 모두 mm 단위.
        물리 스테이지 내부 변환(mm → μm)은 DLL이 담당.

        1. get_position() → 현재 중점 (mm)
        2. SAM 이미지(1060×800) 픽셀 중심 = 스테이지 현재 중점
        3. 각 객체의 픽셀 오프셋 → μm → mm 변환 후 절대좌표 계산

        Args:
            detected_objects: SAM3Segmenter.segment() 반환값

        Returns:
            각 객체에 'stage_x_mm', 'stage_y_mm' 키가 추가
            스테이지 연결 실패 시 None 반환.
        """
        pos = self.stage.get_position()
        if pos is None:
            print("[ERROR] 스테이지 위치 조회 실패")
            return None

        stage_cx_mm, stage_cy_mm = pos[0], pos[1]  # mm (DLL 단위)
        um_per_px_x = LENS_WIDTH  / STREAM_WIDTH    # 304/1060 ≈ 0.2868 μm/px
        um_per_px_y = LENS_HEIGHT / STREAM_HEIGHT   # 230/800  = 0.2875 μm/px

        result = []
        for obj in detected_objects:
            # 이미지 중심 기준 픽셀 오프셋
            dx_px = obj["center_x"] - STREAM_WIDTH  / 2.0
            dy_px = obj["center_y"] - STREAM_HEIGHT / 2.0

            # px → μm → mm (DLL은 mm 단위)
            dx_mm = dx_px * um_per_px_x / 1000.0 * SIGN_X
            dy_mm = dy_px * um_per_px_y / 1000.0 * SIGN_Y

            result.append({
                **obj,
                "stage_x_mm": stage_cx_mm + dx_mm,
                "stage_y_mm": stage_cy_mm + dy_mm,
            })

        return result

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
    t = USE_scan()

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
