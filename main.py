import cv2
import numpy as np
import pyautogui
import pydirectinput
import time
import pytesseract
import sys
import threading
import tkinter as tk
import os
import requests
import subprocess
from tkinter import ttk, messagebox, simpledialog
from windowcapture import WindowCapture

# --- CONFIGURATION ---
CURRENT_VERSION = "1.0.3"
# REPLACE THIS with your GitHub username and repo name
# Example: "MyUser/LKB_Autofishing"
GITHUB_REPO = "DaTaFos/LKB_Autofishing" 
# URLs for updating
VERSION_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/refs/heads/main/version.txt"
EXE_DOWNLOAD_URL = f"https://github.com/{GITHUB_REPO}/releases/latest/download/AutoFishingBot.exe"
# License System
# IMPORTANT: This should be a link to a raw text file containing valid keys (one per line)
# You should host this on a private Gist, a web server, or a private repo (with token in URL)
# For now, it points to the file in your repo (Publicly visible keys! Use with caution!)
LICENSE_KEYS_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/refs/heads/main/keys.txt"
LICENSE_FILE = "license.dat"

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def verify_license(root):
    """ Verifies the license key before starting the app """
    if not getattr(sys, 'frozen', False):
        # Optional: Skip check in dev mode
        # return True 
        pass

    stored_key = ""
    if os.path.exists(LICENSE_FILE):
        try:
            with open(LICENSE_FILE, "r") as f:
                stored_key = f.read().strip()
        except:
            pass

    if stored_key:
        print(f"Verifying saved key: {stored_key}...")
        if check_key_online(stored_key):
             return True
        else:
             messagebox.showerror("License Error", "Your saved license key is invalid or has been revoked.")
             if os.path.exists(LICENSE_FILE):
                 os.remove(LICENSE_FILE)
    
    # Prompt for key
    while True:
        key = tk.simpledialog.askstring("License Verification", "Please enter your license key to continue:", parent=root)
        if not key:
            return False # User cancelled
        
        if check_key_online(key):
            messagebox.showinfo("Success", "License verified! Thank you for purchasing.")
            try:
                with open(LICENSE_FILE, "w") as f:
                    f.write(key.strip())
            except:
                pass
            return True
        else:
            messagebox.showerror("Invalid Key", "That key is not valid. Please contact the seller.")

def check_key_online(key):
    """ Checks if the key exists in the online list """
    try:
        response = requests.get(LICENSE_KEYS_URL, timeout=10)
        if response.status_code == 200:
            valid_keys = response.text.splitlines()
            # Remove whitespace and whitespace-only lines
            valid_keys = [k.strip() for k in valid_keys if k.strip()]
            return key.strip() in valid_keys
    except Exception as e:
        print(f"License check failed: {e}")
        # Optional: Fail open or closed?
        # messagebox.showwarning("Connection Error", "Could not verify license. Please check internet.")
        return False
    return False

def cleanup_old_updates():
    """ Try to remove the old executable if it exists from a previous update """
    if getattr(sys, 'frozen', False):
        exe_path = sys.executable
        old_exe = exe_path + ".old"
        if os.path.exists(old_exe):
            try:
                os.remove(old_exe)
                print("Cleaned up old version.")
            except:
                pass

def check_for_updates(root_window):
    """ Checks for updates and performs self-update if available """
    if not getattr(sys, 'frozen', False):
         print("Running from source, skipping update check.")
         return

    try:
        print(f"Checking for updates at {VERSION_URL}...")
        response = requests.get(VERSION_URL, timeout=5)
        
        if response.status_code == 200:
            remote_version = response.text.strip()
            print(f"Current: {CURRENT_VERSION}, Remote: {remote_version}")
            
            if remote_version != CURRENT_VERSION:
                if messagebox.askyesno("Update Available", f"New version {remote_version} is available!\nUpdate now?"):
                    perform_update(root_window)
    except Exception as e:
        print(f"Update check failed: {e}")

def perform_update(root_window):
    """ Downloads the new exe, swaps files, and restarts """
    exe_path = sys.executable
    
    # CRITICAL: Prevent overwriting python.exe if running from source!
    if "python.exe" in exe_path.lower() or "pythonw.exe" in exe_path.lower():
        messagebox.showerror("Update Error", "Cannot update when running from source (python.exe).\nPlease run the built .exe file to test updates.")
        return

    old_exe = exe_path + ".old"
    
    # Create Update/Progress Window
    update_win = tk.Toplevel(root_window)
    update_win.title("Updating...")
    update_win.geometry("300x120")
    update_win.resizable(False, False)
    
    # Center the window
    x = root_window.winfo_x() + 50
    y = root_window.winfo_y() + 50
    update_win.geometry(f"+{x}+{y}")

    lbl_status = ttk.Label(update_win, text="Starting download...", anchor="center")
    lbl_status.pack(pady=10)
    
    progress = ttk.Progressbar(update_win, orient="horizontal", length=250, mode="determinate")
    progress.pack(pady=10)
    
    update_win.update()

    try:
        print(f"Downloading from {EXE_DOWNLOAD_URL}...")
        
        # 1. Download new file with progress
        r = requests.get(EXE_DOWNLOAD_URL, stream=True)
        r.raise_for_status()
        
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192 # 8KB
        downloaded = 0
        new_data = bytearray()
        
        for chunk in r.iter_content(chunk_size=block_size):
            if chunk:
                new_data.extend(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    progress['value'] = percent
                    lbl_status.config(text=f"Downloading: {int(percent)}% ({downloaded//1024} KB)")
                else:
                    lbl_status.config(text=f"Downloading: {downloaded//1024} KB")
                update_win.update()

        lbl_status.config(text="Installing update...")
        update_win.update()
        
        # 2. Rename current exe to .old (Windows allows renaming running files)
        if os.path.exists(old_exe):
            os.remove(old_exe)
        os.rename(exe_path, old_exe)
        
        # 3. Write new exe
        with open(exe_path, 'wb') as f:
            f.write(new_data)
            
        messagebox.showinfo("Update Success", "Application updated!")
        
        # 4. Restart
        os._exit(0) # Forcefully exit all threads
        
    except Exception as e:
        # If failed, try to restore
        if os.path.exists(old_exe) and not os.path.exists(exe_path):
            os.rename(old_exe, exe_path)
        update_win.destroy()
        messagebox.showerror("Update Failed", f"Failed to update: {e}")

# Configure pytesseract path

# Configure pytesseract path
# 1. Check for bundled Tesseract (for portable exe)
bundled_tesseract = resource_path(os.path.join("Tesseract-OCR", "tesseract.exe"))
if os.path.exists(bundled_tesseract):
    pytesseract.pytesseract.tesseract_cmd = bundled_tesseract
else:
    # 2. Fallback to system install (for development)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# GAME WINDOW NAME
WINDOW_NAME = "FiveM"

class FishingBot:
    def __init__(self):
        self.active = False
        self.debug_mode = False # Production: Default to False
        self.simulation_mode = False
        self.fishing_state = "IDLE"
        self.fps = 0
        self.stop_event = threading.Event()
        
        # Bot State
        self.cached_circle = None
        self.cached_letter = "?"
        self.last_circle_time = time.time()
        self.cooldown_start_time = 0
        self.COOLDOWN_DURATION = 11.5
        self.remaining_cooldown = 0
        
        # Frequencies
        self.frame_count = 0
        self.CIRCLE_DETECT_FREQ = 30
        self.OCR_FREQ = 10
        
        try:
            self.wincap = WindowCapture(WINDOW_NAME)
        except Exception as e:
            print(f"Error initializing WindowCapture: {e}")
            self.wincap = None

    # ... (methods remain the same) ...

    def find_fishing_circle(self, screen_img):
        gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=20, maxRadius=150)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0, 0]
        return None

    def get_angle(self, center, point):
        x, y = point
        cx, cy = center
        return np.arctan2(float(y) - float(cy), float(x) - float(cx)) * 180 / np.pi

    def run(self):
        loop_time = time.time()
        
        while not self.stop_event.is_set():
            if not self.active:
                time.sleep(0.1)
                if self.debug_mode:
                    cv2.destroyAllWindows()
                continue
                
            if self.wincap is None:
                try:
                    self.wincap = WindowCapture(WINDOW_NAME)
                except:
                    time.sleep(1)
                    continue

            try:
                screen_bgr = self.wincap.get_screenshot()
            except Exception as e:
                # print(f"Screenshot error: {e}") # Suppress console errors in prod
                continue

            # FPS Calculation
            current_time = time.time()
            self.fps = 1 / (current_time - loop_time) if current_time - loop_time > 0 else 0
            loop_time = current_time
            self.frame_count += 1

            # 1. Circle Detection
            if self.cached_circle is None or self.frame_count % self.CIRCLE_DETECT_FREQ == 0:
                h, w = screen_bgr.shape[:2]
                margin_x = int(w * 0.25)
                margin_y = int(h * 0.25)
                
                found_new_circle = False
                if margin_x > 0 and margin_y > 0:
                    search_img = screen_bgr[margin_y:h-margin_y, margin_x:w-margin_x]
                    circle = self.find_fishing_circle(search_img)
                    if circle is not None:
                        local_cx, local_cy, r = circle
                        global_cx = int(local_cx + margin_x)
                        global_cy = int(local_cy + margin_y)
                        self.cached_circle = [global_cx, global_cy, int(r)]
                        found_new_circle = True
                
                if not found_new_circle:
                    self.cached_circle = None

            # State Machine
            if self.cached_circle is not None:
                self.last_circle_time = time.time()
                if self.fishing_state != "FISHING":
                    self.fishing_state = "FISHING"
            else:
                if self.fishing_state == "FISHING":
                    if time.time() - self.last_circle_time > 2.0:
                        self.fishing_state = "COOLDOWN"
                        self.cooldown_start_time = time.time()
                elif self.fishing_state == "COOLDOWN":
                    self.remaining_cooldown = self.COOLDOWN_DURATION - (time.time() - self.cooldown_start_time)
                    if self.remaining_cooldown <= 0:
                        if not self.simulation_mode:
                            pydirectinput.press('1')
                        self.fishing_state = "SEARCHING"
                        self.last_circle_time = time.time()

            # Debug Drawing & Logic
            debug_img = None
            if self.debug_mode:
                debug_img = screen_bgr.copy()
                status_color = (0, 255, 0) if self.fishing_state == "FISHING" else (0, 0, 255)
                cv2.putText(debug_img, f"State: {self.fishing_state}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                if self.fishing_state == "COOLDOWN":
                    cv2.putText(debug_img, f"Wait: {int(self.remaining_cooldown)}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                h, w = screen_bgr.shape[:2]
                cv2.rectangle(debug_img, (int(w*0.25), int(h*0.25)), (int(w*0.75), int(h*0.75)), (255, 0, 0), 1)
                cv2.putText(debug_img, f"FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.cached_circle is not None:
                cx, cy, r = self.cached_circle
                if self.debug_mode:
                    cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2)

                # ROI Extraction
                pad = int(r * 0.6)
                x1 = max(0, cx - r - pad)
                y1 = max(0, cy - r - pad)
                x2 = min(screen_bgr.shape[1], cx + r + pad)
                y2 = min(screen_bgr.shape[0], cy + r + pad)
                roi = screen_bgr[y1:y2, x1:x2]
                
                if roi.size > 0:
                    roi_cx = cx - x1
                    roi_cy = cy - y1
                    
                    # OCR
                    if self.frame_count % self.OCR_FREQ == 0:
                        center_r = int(r * 0.5)
                        center_roi = roi[roi_cy-center_r:roi_cy+center_r, roi_cx-center_r:roi_cx+center_r]
                        gray_center = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
                        _, thresh_temp = cv2.threshold(gray_center, 150, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            pad_c = 5
                            x_c = max(0, x - pad_c)
                            y_c = max(0, y - pad_c)
                            w_c = min(thresh_temp.shape[1] - x_c, w + 2*pad_c)
                            h_c = min(thresh_temp.shape[0] - y_c, h + 2*pad_c)
                            letter_crop = thresh_temp[y_c:y_c+h_c, x_c:x_c+w_c]
                            thresh_center = cv2.bitwise_not(letter_crop)
                            
                            target_h = 60
                            scale = target_h / h_c
                            target_w = int(w_c * scale)
                            thresh_center = cv2.resize(thresh_center, (target_w, target_h))
                            thresh_center = cv2.copyMakeBorder(thresh_center, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                            
                            if self.debug_mode:
                                cv2.imshow("OCR Input", thresh_center)

                            found_text = None
                            for psm in [10]:
                                config = f"--psm {psm} -c tessedit_char_whitelist=WASDQER"
                                try:
                                    text = pytesseract.image_to_string(thresh_center, config=config).strip()
                                    if text and text in ['W', 'A', 'S', 'D', 'Q', 'E', 'R']:
                                        found_text = text
                                        break
                                except:
                                    pass
                            if found_text:
                                self.cached_letter = found_text

                    if self.debug_mode:
                        cv2.putText(debug_img, f"Letter: {self.cached_letter}", (cx - r, cy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Red/White Detection
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_ring_area = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.circle(mask_ring_area, (roi_cx, roi_cy), int(r * 1.3), 255, -1)
                    cv2.circle(mask_ring_area, (roi_cx, roi_cy), int(r * 0.9), 0, -1)
                    
                    lower_red1 = np.array([0, 100, 100])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 100, 100])
                    upper_red2 = np.array([180, 255, 255])
                    mask_red = cv2.inRange(hsv_roi, lower_red1, upper_red1) + cv2.inRange(hsv_roi, lower_red2, upper_red2)
                    mask_red = cv2.bitwise_and(mask_red, mask_red, mask=mask_ring_area)
                    
                    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    red_angle = None
                    rx, ry = 0, 0
                    if contours_red:
                        c = max(contours_red, key=cv2.contourArea)
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            rx = int(M["m10"] / M["m00"])
                            ry = int(M["m01"] / M["m00"])
                            red_angle = self.get_angle((roi_cx, roi_cy), (rx, ry))
                            if self.debug_mode:
                                cv2.circle(debug_img, (x1 + rx, y1 + ry), 5, (0, 0, 255), -1)

                    lower_white = np.array([0, 0, 200])
                    upper_white = np.array([180, 50, 255])
                    mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
                    mask_white = cv2.bitwise_and(mask_white, mask_white, mask=mask_ring_area)
                    kernel = np.ones((3,3), np.uint8)
                    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)
                    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    target_found = False
                    if contours_white and red_angle is not None:
                        for wc in contours_white:
                            if cv2.contourArea(wc) < 50: continue
                            if cv2.pointPolygonTest(wc, (rx, ry), True) > -10:
                                target_found = True
                                if self.debug_mode:
                                    cv2.drawContours(debug_img, [wc + (x1, y1)], -1, (255, 0, 0), 2)
                                break
                    
                    if target_found and self.cached_letter:
                        if self.debug_mode:
                            cv2.putText(debug_img, "MATCH!", (cx, cy + r + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        if not self.simulation_mode:
                            time.sleep(0.2)
                            pydirectinput.press(self.cached_letter.lower())

            if self.debug_mode and debug_img is not None:
                cv2.imshow('Fishing Debug', debug_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()

class FishingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Fishing Bot")
        self.root.geometry("250x250") # Compact size
        self.root.resizable(False, False)

        # Version
        self.version = CURRENT_VERSION
        
        try:
            self.root.iconbitmap(resource_path("icon.ico"))
        except Exception as e:
            pass # Icon not found or error loading, ignore
        
        self.bot = FishingBot()
        self.bot_thread = threading.Thread(target=self.bot.run)
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        # Styles
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
        # UI Elements
        self.status_label = ttk.Label(root, text="Status: STOPPED", font=("Arial", 12, "bold"), foreground="red")
        self.status_label.pack(pady=15)
        
        self.btn_toggle = ttk.Button(root, text="START", command=self.toggle_bot)
        self.btn_toggle.pack(pady=5, fill="x", padx=30)
        
        self.info_label = ttk.Label(root, text="Press '1' to cast rod manually first", justify="center", font=("Arial", 8))
        self.info_label.pack(pady=10)

        self.version_label = ttk.Label(root, text=f"Version: {self.version}", justify="center", font=("Arial", 8))
        self.version_label.pack(pady=10)
        
        self.update_ui()
        
        # Start update check in background
        threading.Thread(target=check_for_updates, args=(self.root,), daemon=True).start()

    def toggle_bot(self):
        if not self.bot.active: # Turning ON
            try:
                # Check for game window before starting
                WindowCapture(WINDOW_NAME)
            except Exception:
                messagebox.showerror("Game Not Found", f"Could not find window: '{WINDOW_NAME}'\nPlease ensure the game is running.")
                return

            self.bot.active = True
            self.btn_toggle.config(text="STOP")
            self.status_label.config(text="Status: RUNNING", foreground="green")
        else: # Turning OFF
            self.bot.active = False
            self.btn_toggle.config(text="START")
            self.status_label.config(text="Status: STOPPED", foreground="red")

    def update_ui(self):
        # Update status from bot state if needed
        self.root.after(100, self.update_ui)

if __name__ == "__main__":
    cleanup_old_updates()
    root = tk.Tk()
    
    # Hide the main window initially (optional, but looks cleaner during license check)
    root.withdraw()
    
    if verify_license(root):
        root.deiconify() # Show window if license valid
        app = FishingApp(root)
        root.mainloop()
    else:
        root.destroy()
        sys.exit()
