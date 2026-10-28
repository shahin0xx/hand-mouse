"""
ماوس با دست - نسخه‌ی ۳
نیازمندی‌ها:
    pip install opencv-python mediapipe==0.10.14 pyautogui

ژست‌ها:
    - حرکت دست                      : حرکت ماوس
    - زدن شست به انگشت اشاره        : کلیک چپ  (دو بار سریع = دابل کلیک)
    - زدن شست به انگشت میانی        : کلیک راست
    - مشت کردن دست                  : گرفتن (شروع درگ)
    - باز کردن دست                  : رها کردن (پایان درگ)
    - اشاره + میانی بالا (بقیه بسته): اسکرول (دست بالا = اسکرول بالا)

کلیدها:
    P = توقف/ادامه     M = عوض کردن حالت (relative / absolute)     Esc یا Q = خروج

حالت relative (پیش‌فرض): مثل تاچ‌پد. حرکت آهسته = دقیق، حرکت سریع = پرش بلند.
    با چند حرکت پشت سر هم به هر جای صفحه می‌رسی و لرزش خیلی کمتره.
حالت absolute: جای دست در کادر صورتی = جای ماوس روی صفحه.
"""

import math
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui

# ============================ تنظیمات ============================
CAM_INDEX = None         # None = خودکار (دوربین USB خارجی رو ترجیح می‌ده). یا یه عدد مثل 2
                         # می‌تونی موقع اجرا هم بدی:  python3 vip.py 2
CAM_W, CAM_H = 640, 480

MODE = "relative"        # "relative" یا "absolute"

# --- فیلتر لرزش ---
MIN_CUTOFF = 0.30        # کمتر = نرم‌تر (لرزش کمتر، کمی تأخیر بیشتر)
BETA = 0.006             # بیشتر = تأخیر کمتر در حرکت سریع

# --- حالت relative ---
GAIN_SLOW = 1.2          # ضریب حرکت آهسته (کمتر = دقیق‌تر)
GAIN_FAST = 5.0          # ضریب حرکت سریع (بیشتر = راحت‌تر به لبه‌ها می‌رسی)
SPEED_DEAD = 50          # سرعت (پیکسل/ثانیه) کمتر از این = بی‌حرکت (حذف لرزش)
SPEED_SLOW = 150         # تا این سرعت ضریب = GAIN_SLOW
SPEED_FAST = 1500        # از این سرعت به بالا ضریب = GAIN_FAST

# --- حالت absolute ---
MARGIN = 0.08            # حاشیه‌ی کادر (0 = کل تصویر). بزرگ‌تر = حساس‌تر
DEAD_ZONE = 5            # حرکت کمتر از این (پیکسل) نادیده گرفته می‌شه

# --- کلیک ---
PINCH_ON = 0.25          # شروع کلیک (کمتر = باید انگشت‌ها بیشتر بهم نزدیک بشن)
PINCH_OFF = 0.38         # رها شدن کلیک (کمتر = دابل کلیک سریع‌تر ممکنه)
CLICK_COOLDOWN = 0.12    # حداقل فاصله‌ی بین دو کلیک (ثانیه)
DOUBLE_WINDOW = 0.80     # دو بار زدن شست و اشاره در این مدت (ثانیه) = دابل کلیک
DOUBLE_FAST = 0.35       # اگه فاصله‌ی دو کلیک کمتر از این بود فقط یه کلیک دوم زده می‌شه
REWIND_SEC = 0.20        # کلیک/گرفتن روی مکانی که این‌قدر ثانیه قبل بودی انجام می‌شه
FREEZE_SEC = 0.25        # بعد از کلیک، ماوس این مدت ثابت می‌مونه

# --- مشت ---
FIST_ON_CURLED = 4       # چند انگشت بسته باشه = مشت
FIST_OFF_CURLED = 2      # با کمتر از این تعداد انگشت بسته = دست باز
FIST_MIN_PINCH = 0.20    # اگه شست و اشاره این‌قدر بهم چسبیده باشن مشت حساب نمی‌شه (کلیکه)
CURL_RATIO = 1.2         # نسبت فاصله‌ی نوک انگشت به بند پایینی (کمتر = بسته)

# --- اسکرول ---
ENABLE_SCROLL = True
SCROLL_GAIN = 1200

LOST_GRACE_FRAMES = 8    # چند فریم بدون دست قبل از رها کردن همه‌چیز
DEBUG_TEXT = True        # نمایش اعداد کلیک روی تصویر برای تنظیم
# =================================================================

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()


class OneEuro:
    """فیلتر One Euro: حرکت آهسته = نرم، حرکت سریع = بدون تأخیر."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * abs(dx_hat), dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class Debounce:
    """تغییر وضعیت فقط وقتی چند فریم پشت سر هم ثابت بمونه (حذف خطاهای لحظه‌ای)."""

    def __init__(self, on_frames=2, off_frames=2):
        self.on_frames, self.off_frames = on_frames, off_frames
        self.state = False
        self.count = 0

    def update(self, raw):
        if raw == self.state:
            self.count = 0
        else:
            self.count += 1
            need = self.on_frames if raw else self.off_frames
            if self.count >= need:
                self.state = raw
                self.count = 0
        return self.state

    def reset(self):
        self.state = False
        self.count = 0


def dist(p, q, w, h):
    return math.hypot((p.x - q.x) * w, (p.y - q.y) * h)


def finger_is_curled(lm, tip, mcp, w, h):
    return dist(lm[tip], lm[0], w, h) < CURL_RATIO * dist(lm[mcp], lm[0], w, h)


def curled_count(lm, w, h):
    return sum(finger_is_curled(lm, t, m, w, h)
               for t, m in ((8, 5), (12, 9), (16, 13), (20, 17)))


def open_cam(index):
    """باز کردن دوربین؛ اگه کار نکرد None برمی‌گردونه."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # فریم‌ریت بهتر روی دوربین USB
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def find_cameras(max_index=8):
    """شماره‌ی دوربین‌هایی که واقعاً تصویر می‌دن."""
    found = []
    for i in range(max_index):
        cap = open_cam(i)
        if cap is not None:
            found.append(i)
            cap.release()
    return found


def gain_for_speed(speed):
    if speed < SPEED_DEAD:
        return 0.0
    k = (speed - SPEED_SLOW) / max(SPEED_FAST - SPEED_SLOW, 1.0)
    k = min(1.0, max(0.0, k))
    return GAIN_SLOW + (GAIN_FAST - GAIN_SLOW) * k


def main():
    global MODE

    print("Screen size seen by pyautogui: %dx%d" % (screen_w, screen_h))

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,          # دقیق‌تر = لرزش کمتر (کمی کندتر)
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cams = find_cameras()
    print("Available cameras:", cams)
    if not cams:
        print("هیچ دوربینی پیدا نشد. اتصال USB رو چک کن:  ls /dev/video*")
        return

    start = int(sys.argv[1]) if len(sys.argv) > 1 else CAM_INDEX
    if start is None or start not in cams:
        start = cams[-1]          # معمولاً دوربین USB خارجی آخرین شماره‌ست
    cam_pos = cams.index(start)
    print("Using camera:", start, " (کلید C = دوربین بعدی)")
    cap = open_cam(start)
    if cap is None:
        print("دوربین باز نشد.")
        return

    fx = OneEuro(MIN_CUTOFF, BETA)
    fy = OneEuro(MIN_CUTOFF, BETA)

    history = deque(maxlen=90)  # (زمان, x, y)
    cur_x, cur_y = pyautogui.position()
    prev_fx = prev_fy = prev_t = None   # برای حالت relative

    def pos_back(now, sec):
        target = now - sec
        for t, x, y in reversed(history):
            if t <= target:
                return x, y
        if history:
            return history[0][1], history[0][2]
        return cur_x, cur_y

    def move_to(x, y):
        nonlocal cur_x, cur_y
        cur_x = min(screen_w - 2, max(1, x))
        cur_y = min(screen_h - 2, max(1, y))
        pyautogui.moveTo(cur_x, cur_y)

    fist = Debounce(on_frames=3, off_frames=3)
    pinch_l = Debounce(on_frames=2, off_frames=2)
    pinch_r = Debounce(on_frames=2, off_frames=2)
    scroll_d = Debounce(on_frames=5, off_frames=2)

    paused = False
    dragging = False
    last_click = 0.0
    last_left_t = -10.0
    last_left_pos = None
    freeze_until = 0.0
    lost = 0
    scroll_prev_y = None
    scroll_acc = 0.0

    def release_all():
        nonlocal dragging
        if dragging:
            pyautogui.mouseUp()
            dragging = False
        for d in (fist, pinch_l, pinch_r, scroll_d):
            d.reset()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.perf_counter()

        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if MODE == "absolute":
            x0, y0 = int(MARGIN * w), int(MARGIN * h)
            x1, y1 = int((1 - MARGIN) * w), int((1 - MARGIN) * h)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 255), 2)

        status = "PAUSED" if paused else "MOVE"
        dbg = ""

        if result.multi_hand_landmarks:
            lost = 0
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            size = max(dist(lm[0], lm[9], w, h), 1.0)

            # مرکز کف دست: پایدارتر از نوک انگشت
            ids = (0, 5, 9, 13, 17)
            px = sum(lm[i].x for i in ids) / len(ids)
            py = sum(lm[i].y for i in ids) / len(ids)
            cv2.circle(frame, (int(px * w), int(py * h)), 10, (0, 255, 0), -1)

            # موقعیت فیلترشده (مقیاس: کل تصویر = کل صفحه)
            if MODE == "absolute":
                m = MARGIN
                nx = min(1.0, max(0.0, (px - m) / (1 - 2 * m)))
                ny = min(1.0, max(0.0, (py - m) / (1 - 2 * m)))
            else:
                nx, ny = px, py
            tx = fx(nx * (screen_w - 1), now)
            ty = fy(ny * (screen_h - 1), now)

            # اولین فریم بعد از ظاهر شدن دست: ماوس رو با جای واقعی هماهنگ کن
            if prev_fx is None:
                cur_x, cur_y = pyautogui.position()
                prev_fx, prev_fy, prev_t = tx, ty, now

            # تغییر موقعیت این فریم (برای حالت relative)
            dtm = max(now - prev_t, 1e-3)
            ddx, ddy = tx - prev_fx, ty - prev_fy
            gain = gain_for_speed(math.hypot(ddx, ddy) / dtm)
            prev_fx, prev_fy, prev_t = tx, ty, now

            rl = dist(lm[4], lm[8], w, h) / size    # شست - اشاره
            rr = dist(lm[4], lm[12], w, h) / size   # شست - میانی
            n_curled = curled_count(lm, w, h)
            dbg = "L:%.2f R:%.2f curled:%d" % (rl, rr, n_curled)

            scrolling = False

            if not paused:
                # ---------- مشت = گرفتن / درگ ----------
                if fist.state:
                    raw_fist = n_curled >= FIST_OFF_CURLED + 1
                else:
                    raw_fist = n_curled >= FIST_ON_CURLED and rl > FIST_MIN_PINCH
                was_fist = fist.state
                is_fist = fist.update(raw_fist)

                if is_fist and not was_fist:
                    gx, gy = pos_back(now, REWIND_SEC)
                    move_to(gx, gy)
                    pyautogui.mouseDown()
                    dragging = True
                elif was_fist and not is_fist:
                    gx, gy = pos_back(now, REWIND_SEC)
                    move_to(gx, gy)
                    pyautogui.mouseUp()
                    dragging = False
                    freeze_until = now + FREEZE_SEC

                if is_fist:
                    status = "DRAG"
                    pinch_l.reset()
                    pinch_r.reset()
                else:
                    # ---------- کلیک چپ / راست ----------
                    # هر کدوم از دو جفت انگشت که نزدیک‌تره برنده‌ست
                    # (وقتی شست به اشاره می‌خوره، به میانی هم نزدیکه؛ قبلاً همین کلیک چپ رو خراب می‌کرد)
                    cand_left = rl <= rr
                    was_l, was_r = pinch_l.state, pinch_r.state
                    on_l = pinch_l.update(cand_left and rl < (PINCH_OFF if was_l else PINCH_ON))
                    on_r = pinch_r.update((not cand_left) and rr < (PINCH_OFF if was_r else PINCH_ON))

                    if on_l and not was_l and now - last_click > CLICK_COOLDOWN:
                        gap = now - last_left_t
                        if gap < DOUBLE_WINDOW and last_left_pos is not None:
                            # کلیک دوم: روی همون نقطه‌ی کلیک اول
                            move_to(*last_left_pos)
                            if gap < DOUBLE_FAST:
                                pyautogui.click()
                            else:
                                pyautogui.doubleClick()
                            last_left_t = -10.0   # کلیک سوم دوباره از اول حساب می‌شه
                            status = "DOUBLE CLICK"
                        else:
                            gx, gy = pos_back(now, REWIND_SEC)
                            move_to(gx, gy)
                            pyautogui.click()
                            last_left_t = now
                            last_left_pos = (cur_x, cur_y)
                        last_click = now
                        freeze_until = now + FREEZE_SEC
                    elif on_r and not was_r and now - last_click > CLICK_COOLDOWN:
                        gx, gy = pos_back(now, REWIND_SEC)
                        move_to(gx, gy)
                        pyautogui.click(button="right")
                        last_click = now
                        freeze_until = now + FREEZE_SEC

                    if on_l:
                        status = "LEFT CLICK"
                    elif on_r:
                        status = "RIGHT CLICK"

                    # ---------- اسکرول ----------
                    if ENABLE_SCROLL:
                        raw_scroll = (
                            not finger_is_curled(lm, 8, 5, w, h)
                            and not finger_is_curled(lm, 12, 9, w, h)
                            and finger_is_curled(lm, 16, 13, w, h)
                            and finger_is_curled(lm, 20, 17, w, h)
                            and rl > PINCH_OFF and rr > PINCH_OFF
                        )
                        scrolling = scroll_d.update(raw_scroll)
                        if scrolling:
                            status = "SCROLL"
                            if scroll_prev_y is not None:
                                scroll_acc += (scroll_prev_y - py) * SCROLL_GAIN
                                steps = int(scroll_acc)
                                if steps != 0:
                                    pyautogui.scroll(steps)
                                    scroll_acc -= steps
                            scroll_prev_y = py
                    if not scrolling:
                        scroll_prev_y = None
                        scroll_acc = 0.0

                # ---------- حرکت ماوس ----------
                if not scrolling and now >= freeze_until:
                    if MODE == "relative":
                        if gain > 0:
                            move_to(cur_x + ddx * gain, cur_y + ddy * gain)
                    else:
                        if math.hypot(tx - cur_x, ty - cur_y) > DEAD_ZONE:
                            move_to(tx, ty)

            history.append((now, cur_x, cur_y))
        else:
            lost += 1
            status = "NO HAND"
            if lost > LOST_GRACE_FRAMES:
                release_all()
                fx.reset()
                fy.reset()
                prev_fx = prev_fy = prev_t = None
                history.clear()
                scroll_prev_y = None
                scroll_acc = 0.0

        cv2.putText(frame, "%s | %s" % (status, MODE), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if DEBUG_TEXT and dbg:
            cv2.putText(frame, dbg, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
        cv2.imshow("Hand Mouse  (P pause, M mode, C camera, Esc exit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            paused = not paused
            release_all()
            prev_fx = None
        if key == ord("c") and len(cams) > 1:
            release_all()
            cap.release()
            cam_pos = (cam_pos + 1) % len(cams)
            cap = open_cam(cams[cam_pos])
            if cap is None:
                print("دوربین باز نشد.")
                break
            print("Using camera:", cams[cam_pos])
            fx.reset()
            fy.reset()
            prev_fx = None
            history.clear()
        if key == ord("m"):
            MODE = "absolute" if MODE == "relative" else "relative"
            release_all()
            fx.reset()
            fy.reset()
            prev_fx = None
            history.clear()

    release_all()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()