import os
os.environ["PYAV_LOGGING"] = "off"
import sys
import time
import gc
import shutil
import base64
import subprocess
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty as QueueEmpty

import cv2
import numpy as np

try:
    import av
    av.logging.disable()
except Exception:
    pass

from adbutils import adb


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # 开发环境：用脚本所在目录，而不是 CWD
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def find_adb():
    candidates = [
        resource_path(os.path.join('adb', 'adb.exe')),
        resource_path(os.path.join('adb', 'adb')),
        shutil.which('adb') or '',
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


LOCAL_ADB = find_adb()

# ========== 参数 ==========
THRESHOLD = 0.8
TAP_COOLDOWN = 0.8
CHECK_INTERVAL = 0.02
STALL_TIMEOUT = 15.0
WORKER_STALL_SEC = 3.0
MAX_FPS = 15
ORIENTATION_CHECK_INTERVAL = 1.5

# 匹配刷新间隔（matcher 匹配一次后的最短等待）
MATCH_INTERVAL = 0.1

# 组合点击：icon2 → 等待 → icon3
COMBO_CLICK_DELAY = 0.3


# ============================================================
# worker：mysc-core 解码，原始分辨率写入共享内存
# ============================================================
def video_worker(serial, out_w, out_h, shm_name, seq_val):
    from adbutils import adb as adb_module
    from mysc_core.video import VideoAdapter, VideoKwargs

    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((out_h, out_w, 3), dtype=np.uint8, buffer=shm.buf)

    while True:
        va = None
        try:
            device = None
            for d in adb_module.device_list():
                if d.serial == serial:
                    device = d
                    break
            if device is None:
                time.sleep(2)
                continue

            # 完全等同原版
            kwargs = VideoKwargs(
                video_codec=VideoKwargs.EnumVideoCodec.H264,
                max_fps=MAX_FPS
            )
            va = VideoAdapter(kwargs)
            va.connect(device)
            print(f"[worker] 已连接 {out_w}x{out_h}")

            push_count = 0
            last_frame_n = va.frame_n
            last_progress_time = time.time()

            while True:
                current_n = va.frame_n
                if current_n != last_frame_n:
                    last_frame_n = current_n
                    last_progress_time = time.time()
                else:
                    if time.time() - last_progress_time > WORKER_STALL_SEC:
                        print("[worker] frame_n 停滞，重连")
                        break

                frame_obj = va.get_frame()
                if frame_obj is None:
                    time.sleep(0.003)
                    continue

                try:
                    # 原版是 rgb24 再 cvtColor，这里直接 bgr24（等价）
                    frame_bgr = frame_obj.to_ndarray(format='bgr24')
                except Exception as e:
                    print(f"[worker] to_ndarray 失败: {e}")
                    break

                # 方向不匹配时 resize（只发生在切换瞬间）
                if frame_bgr.shape[1] != out_w or frame_bgr.shape[0] != out_h:
                    frame_bgr = cv2.resize(
                        frame_bgr, (out_w, out_h),
                        interpolation=cv2.INTER_AREA
                    )

                # 原始分辨率写入共享内存
                np.copyto(arr, frame_bgr)
                seq_val.value += 1

                push_count += 1
                if push_count == 1:
                    print(f"[worker] 首帧 {out_w}x{out_h}")
                elif push_count % 300 == 0:
                    print(f"[worker] 已推送 {push_count} 帧")

        except Exception as e:
            print(f"[worker] 异常: {e}")

        if va is not None:
            try:
                va.disconnect()
            except Exception:
                pass
            va = None
        gc.collect()
        time.sleep(2)


# ============================================================
# Session
# ============================================================
class Session:
    def __init__(self, device, serial, w, h):
        self.device = device
        self.serial = serial
        self.W = w
        self.H = h

        self.template_path = resource_path('icon.png')
        if not os.path.exists(self.template_path):
            raise RuntimeError("找不到 icon.png")

        self.shm = None
        self.frame_view = None
        self.seq_val = None
        self.proc_worker = None

    def start(self):
        frame_size = self.W * self.H * 3
        self.shm = shared_memory.SharedMemory(create=True, size=frame_size)
        self.frame_view = np.ndarray(
            (self.H, self.W, 3),
            dtype=np.uint8, buffer=self.shm.buf
        )
        self.seq_val = mp.Value('L', 0)

        self.proc_worker = mp.Process(
            target=video_worker,
            args=(self.serial, self.W, self.H,
                  self.shm.name, self.seq_val),
            daemon=True
        )
        self.proc_worker.start()

        # 等首帧
        wait_start = time.time()
        while self.seq_val.value == 0:
            if time.time() - wait_start > 30:
                print("[Session] 首帧超时")
                break
            time.sleep(0.05)
        print(f"[Session] 首帧就绪 {self.W}x{self.H}")

    def stop(self):
        if self.proc_worker is not None:
            try:
                self.proc_worker.terminate()
                self.proc_worker.join(timeout=1)
                if self.proc_worker.is_alive():
                    self.proc_worker.kill()
            except Exception:
                pass
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass
        if LOCAL_ADB:
            try:
                subprocess.run(
                    [LOCAL_ADB, '-s', self.serial, 'shell',
                     'pkill', '-f', 'scrcpy'],
                    timeout=3, capture_output=True
                )
            except Exception:
                pass


# ============================================================
# 匹配（原版逻辑）
# ============================================================
def find_icon(frame_bgr, template, threshold):
    result = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None, max_val
    h, w = template.shape[:2]
    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2
    cv2.rectangle(frame_bgr, max_loc,
                  (max_loc[0] + w, max_loc[1] + h),
                  (0, 255, 0), 2)
    cv2.circle(frame_bgr, (cx, cy), 8, (0, 0, 255), -1)
    return (cx, cy), max_val


def capture_screen(device):
    """仅在启动时用一次，判断初始方向"""
    PHONE_TMP = '/sdcard/_startup.png'
    try:
        device.shell(['screencap', '-p', PHONE_TMP])
        b64_str = device.shell(['base64', PHONE_TMP])
        device.shell(['rm', PHONE_TMP])
        b64_clean = (b64_str.strip()
                     .replace('\n', '')
                     .replace('\r', '')
                     .replace(' ', ''))
        if len(b64_clean) < 1000:
            return None
        img_bytes = base64.b64decode(b64_clean)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # ★ 打包修改：PyInstaller 打包后必须调用，否则 spawn 子进程会无限递归启动主程序
    mp.freeze_support()

    mp.set_start_method('spawn', force=True)

    try:
        device = adb.device_list()[0]
        serial = device.serial
        print(f"设备已连接: {serial}")
    except IndexError:
        print("未检测到设备")
        sys.exit(1)

    # 获取手机分辨率
    try:
        out = device.shell(['wm', 'size']).strip()
        size_str = out.split(':')[-1].strip()
        PHONE_W, PHONE_H = map(int, size_str.split('x'))
    except Exception:
        PHONE_W, PHONE_H = 1080, 2160

    # 判断初始方向（只启动时做一次截图）
    print("启动检测方向...")
    startup = capture_screen(device)
    if startup is not None:
        sh, sw = startup.shape[:2]
        orientation = 'landscape' if sw > sh else 'portrait'
    else:
        orientation = 'portrait'
    print(f"方向: {orientation} 手机: {PHONE_W}x{PHONE_H}")

    if orientation == 'landscape':
        W, H = PHONE_H, PHONE_W
    else:
        W, H = PHONE_W, PHONE_H

    try:
        session = Session(device, serial, W, H)
        session.start()
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

    template = cv2.imread(session.template_path)
    if template is None:
        print("模板读取失败")
        session.stop()
        sys.exit(1)
    print(f"模板: {template.shape[1]}x{template.shape[0]}")

    # ===== 加载 icon2 / icon3 组合模板 =====
    icon2_path = resource_path('icon2.png')
    icon3_path = resource_path('icon3.png')
    template2 = cv2.imread(icon2_path) if os.path.exists(icon2_path) else None
    template3 = cv2.imread(icon3_path) if os.path.exists(icon3_path) else None
    if template2 is None or template3 is None:
        print("icon2.png / icon3.png 缺失，组合点击功能已禁用")
        template2 = None
        template3 = None
    else:
        print(f"组合模板: icon2 {template2.shape[1]}x{template2.shape[0]}, "
              f"icon3 {template3.shape[1]}x{template3.shape[0]}")

    window_name = 'Android Screen'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if orientation == 'landscape':
        cv2.resizeWindow(window_name, 720, 360)
    else:
        cv2.resizeWindow(window_name, 360, 720)

    def tap(x, y):
        cmd = ['input', 'tap', str(x), str(y)]
        try:
            device.shell(cmd)
            print(f"已点击: ({x}, {y})")
            return True
        except Exception as e:
            print(f"点击失败: {e}")
        if LOCAL_ADB:
            try:
                r = subprocess.run(
                    [LOCAL_ADB, '-s', serial, 'shell'] + cmd,
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    print(f"已点击[adb]: ({x}, {y})")
                    return True
            except Exception:
                pass
        return False

    def draw_debug_info(frame, score, found, tap_count, cooldown_left,
                        fps, last_pos, threshold, max_score, orientation):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        color = (0, 255, 0) if found else (0, 165, 255)
        cv2.putText(frame, f"Status: {'FOUND' if found else 'SEARCHING'}",
                    (10, 25), font, scale, color, thickness)
        cv2.putText(frame, f"Score: {score:.3f}", (10, 50),
                    font, scale, (255, 255, 255), thickness)
        cv2.putText(frame, f"Max: {max_score:.3f}  Thresh: {threshold:.2f}",
                    (10, 75), font, scale, (0, 255, 255), thickness)
        cv2.putText(frame, f"Total Taps: {tap_count}", (10, 100),
                    font, scale, (255, 255, 255), thickness)
        if cooldown_left > 0:
            cv2.putText(frame, f"Cooldown: {cooldown_left:.2f}s", (10, 125),
                        font, scale, (0, 0, 255), thickness)
        else:
            cv2.putText(frame, "Cooldown: ready", (10, 125),
                        font, scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150),
                    font, scale, (255, 255, 0), thickness)
        if last_pos:
            cv2.putText(frame, f"Last tap: {last_pos[0]},{last_pos[1]}",
                        (10, 175), font, scale, (200, 200, 200), thickness)
        cv2.putText(frame, f"Orientation: {orientation}",
                    (10, 200), font, scale, (200, 200, 255), thickness)

    last_tap_time = 0
    tap_count = 0
    icon_visible_last = False
    last_pos = None
    fps = 0.0
    fps_timer = time.time()
    frame_count = 0
    max_score = 0.0
    threshold = THRESHOLD
    save_counter = 0
    last_seq = 0
    last_new_frame_time = time.time()
    last_check_time = 0.0
    current_orientation = orientation
    last_frame = None
    cur_pos = None
    cur_score = 0.0

    try:
        while True:
            # 检查 seq 判断是否有新帧
            cur_seq = session.seq_val.value
            if cur_seq != last_seq:
                last_seq = cur_seq
                last_new_frame_time = time.time()
                # 主进程 copy 一份用于显示+匹配
                last_frame = session.frame_view.copy()

            # 方向检测
            now = time.time()
            if now - last_check_time > ORIENTATION_CHECK_INTERVAL:
                last_check_time = now
                if last_frame is not None:
                    h_, w_ = last_frame.shape[:2]
                    detected = 'landscape' if w_ > h_ else 'portrait'
                    if detected != current_orientation:
                        print(f"\n*** 方向变化: {current_orientation} → "
                              f"{detected} ***")
                        try:
                            session.stop()
                        except Exception:
                            pass
                        try:
                            if detected == 'landscape':
                                W2, H2 = PHONE_H, PHONE_W
                            else:
                                W2, H2 = PHONE_W, PHONE_H
                            session = Session(device, serial, W2, H2)
                            session.start()
                            template = cv2.imread(session.template_path)
                            current_orientation = detected
                            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            if detected == 'landscape':
                                cv2.resizeWindow(window_name, 720, 360)
                            else:
                                cv2.resizeWindow(window_name, 360, 720)
                            tap_count = 0
                            max_score = 0.0
                            last_seq = 0
                            last_frame = None
                            last_new_frame_time = time.time()
                            print(f"✓ 已切换到 {detected}")
                        except Exception as e:
                            print(f"✗ 切换失败: {e}")

            # worker 看门狗
            if time.time() - last_new_frame_time > STALL_TIMEOUT:
                print(f"\n[Main] worker 超时，重启")
                try:
                    session.proc_worker.terminate()
                    session.proc_worker.join(timeout=1)
                except Exception:
                    pass
                try:
                    session.proc_worker = mp.Process(
                        target=video_worker,
                        args=(serial, session.W, session.H,
                              session.shm.name, session.seq_val),
                        daemon=True
                    )
                    session.proc_worker.start()
                except Exception as e:
                    print(f"重启失败: {e}")
                last_new_frame_time = time.time()

            if last_frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.01)
                continue

            # ================= 匹配 =================
            display = last_frame.copy()
            now = time.time()
            cooldown_left = max(0, TAP_COOLDOWN - (now - last_tap_time))

            # -------- 优先：icon2 + icon3 组合检测 --------
            combo_active = False
            combo_score = 0.0
            if template2 is not None and template3 is not None:
                pos2, score2 = find_icon(display, template2, threshold)
                pos3, score3 = find_icon(display, template3, threshold)
                if pos2 is not None and pos3 is not None:
                    combo_active = True
                    combo_score = max(score2, score3)
                    if combo_score > max_score:
                        max_score = combo_score
                    if now - last_tap_time >= TAP_COOLDOWN:
                        print(f"\n[Combo] icon2 @{pos2} ({score2:.3f}), "
                              f"icon3 @{pos3} ({score3:.3f})")
                        print("[Combo] 先点击 icon2")
                        tap(pos2[0], pos2[1])
                        time.sleep(COMBO_CLICK_DELAY)
                        print("[Combo] 再点击 icon3")
                        tap(pos3[0], pos3[1])
                        tap_count += 2
                        last_tap_time = time.time()
                        last_pos = pos3
                        cooldown_left = TAP_COOLDOWN

            # -------- 组合未命中：走原来的单图标逻辑 --------
            if combo_active:
                found = True
                score = combo_score
                cur_pos = None
                cur_score = combo_score
            else:
                pos, score = find_icon(display, template, threshold)
                cur_pos = pos
                cur_score = score
                found = pos is not None
                if score > max_score:
                    max_score = score

                if found:
                    icon_visible_last = True
                    if now - last_tap_time >= TAP_COOLDOWN:
                        x, y = pos
                        print(f"\n检测到图标 ({x}, {y})，得分 {score:.3f}，"
                              f"执行第 {tap_count + 1} 次点击")
                        tap(x, y)
                        tap_count += 1
                        last_tap_time = now
                        last_pos = (x, y)
                else:
                    if icon_visible_last:
                        print(f"\n图标已消失（累计点击 {tap_count} 次），"
                              f"继续监测...")
                        icon_visible_last = False

            frame_count += 1
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            draw_debug_info(display, score, found, tap_count,
                            cooldown_left, fps, last_pos, threshold,
                            max_score, current_orientation)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                save_counter += 1
                cv2.imwrite(f'debug_frame_{save_counter}.png', display)
                print(f"\n已保存 debug_frame_{save_counter}.png")
            elif key == ord('t'):
                cx, cy = session.W // 2, session.H // 2
                print(f"\n[T] 强制点击中心 ({cx},{cy})")
                tap(cx, cy)
            elif key in (ord('+'), ord('=')):
                threshold = min(0.99, threshold + 0.02)
                print(f"\n阈值 → {threshold:.2f}")
            elif key in (ord('-'), ord('_')):
                threshold = max(0.30, threshold - 0.02)
                print(f"\n阈值 → {threshold:.2f}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        try:
            session.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("已退出")