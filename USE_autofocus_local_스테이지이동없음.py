"""
가이드빔 + 카메라 스트리밍 통합 로컬 오토포커스 모듈
- LaserController  : USE_laser_with_power.py 의 클래스 재사용
- StreamingTUCam   : USE_camera_stream.py 의 클래스 재사용
ESC 키로 종료
"""

import sys
import os
import cv2
import numpy as np

# 같은 디렉터리의 모듈 임포트
sys.path.append(os.path.dirname(__file__))
from USE_laser_with_power import LaserController
from USE_camera_stream import StreamingTUCam
from USE_stage_test import TangoController

STREAM_WIDTH  = 1060
STREAM_HEIGHT = 800


class AutoFocusLocal:
    """
    가이드빔을 쏘면서 카메라(1060x800)로 실시간 스트리밍하는 클래스.
    레이저/카메라 제어는 각각의 원본 클래스 인스턴스에 위임합니다.
    """

    def __init__(self, laser_port: str = 'COM4', exposure_ms: float = 10.0):
        print("=== AutoFocusLocal 초기화 ===")

        # 레이저 컨트롤러 (USE_laser_with_power.py)
        self.laser = LaserController(port=laser_port)

        # 카메라 스트리머 (USE_camera_stream.py)
        self.camera = StreamingTUCam(exposure_ms=exposure_ms)

        self._exposure_ms = exposure_ms

    # ------------------------------------------------------------------
    # 가이드빔 제어
    # ------------------------------------------------------------------
    def guide_beam_on(self):
        """가이드빔 모드 활성화 후 레이저 ON"""
        self.laser.set_guide_beam()   # 필터를 가이드빔 위치로 이동

    def guide_beam_off(self):
        """레이저 OFF"""
        self.laser.laser_off()

    # ------------------------------------------------------------------
    # 카메라 스트리밍
    # ------------------------------------------------------------------
    def run(self):
        """
        가이드빔을 켜고, 1060x800 해상도로 카메라를 스트리밍합니다.
        ESC  : 종료
        E    : 노출 +5ms
        D    : 노출 -5ms
        """
        print(f"스트리밍 해상도: {STREAM_WIDTH}x{STREAM_HEIGHT}")
        print("ESC: 종료 | E: 노출 증가 | D: 노출 감소")

        # 1. 일단 가이드빔 위치(필터)로 세팅만 해둡니다. (레이저는 아직 안 켬!)
        print("가이드빔 필터 세팅 중...")
        self.laser.set_guide_beam()
        self.laser.laser_on

        # 2. 카메라 스트리밍 시작 (먼저 눈부터 뜹니다)
        self.camera.start_stream()

        try:
            while True:
                # StreamingTUCam.get_latest_frame 호출
                frame = self.camera.get_latest_frame()
                if frame is None:
                    continue

                # 디스플레이용 변환
                disp = frame.copy()
                if disp.dtype == np.uint16:
                    disp = (disp / 256).astype(np.uint8)
                if len(disp.shape) == 2:
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

                # 1060x800 리사이즈
                disp = cv2.resize(disp, (STREAM_WIDTH, STREAM_HEIGHT))

                # 정보 오버레이
                cv2.putText(disp, f"Guide Beam ON | Exposure: {self._exposure_ms:.1f}ms",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imshow("AutoFocus Local - Guide Beam Stream", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:   # ESC
                    break
                elif key in (ord('e'), ord('E')):
                    self._exposure_ms += 5.0
                    self.camera.set_exposure(self._exposure_ms)
                    print(f"Exposure: {self._exposure_ms:.1f}ms")
                elif key in (ord('d'), ord('D')):
                    self._exposure_ms = max(1.0, self._exposure_ms - 5.0)
                    self.camera.set_exposure(self._exposure_ms)
                    print(f"Exposure: {self._exposure_ms:.1f}ms")
                

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    # ------------------------------------------------------------------
    # 정리
    # ------------------------------------------------------------------
    def close(self):
        print("종료 중...")
        self.guide_beam_off()
        self.camera.close()
        cv2.destroyAllWindows()
        self.laser.close()
        print("AutoFocusLocal 종료 완료.")


# ----------------------------------------------------------------------
# 단독 실행 진입점
# ----------------------------------------------------------------------
def main():
    port_input = input("레이저 COM 포트 (기본값: COM4): ").strip()
    port = port_input.upper() if port_input else 'COM4'

    exp_input = input("초기 노출 시간 ms (기본값: 10.0): ").strip()
    try:
        exposure = float(exp_input) if exp_input else 10.0
    except ValueError:
        exposure = 10.0

    af = AutoFocusLocal(laser_port=port, exposure_ms=exposure)
    af.run()


if __name__ == "__main__":
    main()
