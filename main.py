import cv2
import numpy as np
import pyautogui
import pydirectinput

import time
# from PIL import ImageGrab # No longer needed with WindowCapture
import pytesseract
import sys
from windowcapture import WindowCapture



# Configure pytesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# GAME WINDOW NAME
# IMPORTANT: Change this to the exact name of your game window!
WINDOW_NAME = "FiveM" 

def find_fishing_circle(screen_img):
    gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Adjust these parameters based on the actual size of the circle on screen
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=20, maxRadius=150)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Return the strongest circle
        return circles[0, 0]
    return None

def get_angle(center, point):
    # Calculate angle of point relative to center
    x, y = point
    cx, cy = center
    return np.arctan2(float(y) - float(cy), float(x) - float(cx)) * 180 / np.pi

def main():
    print("Starting Auto Fishing Macro...")
    print(f"Looking for window: '{WINDOW_NAME}'")
    print("Press 'q' to quit.")
    print("Press 's' to toggle simulation mode (no key presses).")
    
    # Initialize WindowCapture
    try:
        wincap = WindowCapture(WINDOW_NAME)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the game is running and the WINDOW_NAME in main.py is correct.")
        return

    simulation_mode = False
    
    loop_time = time.time()
    frame_count = 0
    
    # Cache variables
    cached_circle = None
    cached_letter = "?"
    
    # Frequencies (run every N frames)
    CIRCLE_DETECT_FREQ = 30
    OCR_FREQ = 10
    
    # State Machine
    # States: "SEARCHING", "FISHING", "COOLDOWN"
    fishing_state = "SEARCHING"
    last_circle_time = time.time()
    cooldown_start_time = 0
    remaining = int(0)
    COOLDOWN_DURATION = 11.5 # Seconds

    # Debug Mode
    debug_mode = True

    while True:
        # ... (capture screen) ...
        try:
            screen_bgr = wincap.get_screenshot()
        except Exception as e:
            print(f"Screenshot error: {e}")
            continue
            
        # ... (FPS calculation) ...
        current_time = time.time()
        fps = 1 / (current_time - loop_time)
        loop_time = current_time
        frame_count += 1
        
        # ... (Circle Detection) ...
        if cached_circle is None or frame_count % CIRCLE_DETECT_FREQ == 0:
            # ... (detection logic) ...
            # Try center search first
            h, w = screen_bgr.shape[:2]
            margin_x = int(w * 0.25)
            margin_y = int(h * 0.25)
            
            found_new_circle = False
            
            if margin_x > 0 and margin_y > 0:
                search_img = screen_bgr[margin_y:h-margin_y, margin_x:w-margin_x]
                circle = find_fishing_circle(search_img)
                if circle is not None:
                    local_cx, local_cy, r = circle
                    global_cx = int(local_cx + margin_x)
                    global_cy = int(local_cy + margin_y)
                    cached_circle = [global_cx, global_cy, int(r)]
                    found_new_circle = True
            
            # If not found in center, we assume no circle is present
            if not found_new_circle:
                cached_circle = None

        # State Machine Logic
        if cached_circle is not None:
            last_circle_time = time.time()
            if fishing_state != "FISHING":
                fishing_state = "FISHING"
                print("Fishing UI Detected. State: FISHING")
        else:
            # Circle not visible
            if fishing_state == "FISHING":
                # Check if it's really gone (debounce 2s)
                if time.time() - last_circle_time > 2.0:
                    fishing_state = "COOLDOWN"
                    cooldown_start_time = time.time()
                    print(f"Fishing finished. State: COOLDOWN (Waiting {COOLDOWN_DURATION}s)")
            
            elif fishing_state == "COOLDOWN":
                remaining = COOLDOWN_DURATION - (time.time() - cooldown_start_time)
                if remaining <= 0:
                    print("Cooldown finished. Pressing '1' to restart.")
                    if not simulation_mode:
                        pydirectinput.press('1')
                    fishing_state = "SEARCHING"
                    last_circle_time = time.time() # Reset timer
        
        # Prepare debug image only if needed
        if debug_mode:
            debug_img = screen_bgr.copy()
            
            # Draw Status
            status_color = (0, 255, 0) if fishing_state == "FISHING" else (0, 0, 255)
            cv2.putText(debug_img, f"State: {fishing_state}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            if fishing_state == "COOLDOWN":
                 # Calculate remaining time for display if not already calculated
                 if 'remaining' not in locals():
                     remaining = COOLDOWN_DURATION - (time.time() - cooldown_start_time)
                 cv2.putText(debug_img, f"Wait: {int(remaining)}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw search area rectangle for debug
            h, w = screen_bgr.shape[:2]
            cv2.rectangle(debug_img, (int(w*0.25), int(h*0.25)), (int(w*0.75), int(h*0.75)), (255, 0, 0), 1)
            
            cv2.putText(debug_img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cached_circle is not None:
            cx, cy, r = cached_circle
            
            if debug_mode:
                # Draw detected circle
                cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2)
            
            # Extract ROI for processing
            # Increase padding to capture the outer ring (cursor/bar)
            pad = int(r * 0.6) 
            x1 = max(0, cx - r - pad)
            y1 = max(0, cy - r - pad)
            x2 = min(screen_bgr.shape[1], cx + r + pad)
            y2 = min(screen_bgr.shape[0], cy + r + pad)
            
            roi = screen_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                # Coordinates relative to ROI
                roi_cx = cx - x1
                roi_cy = cy - y1
                
                # 2. Detect Letter (Very Expensive)
                # Run only every N frames
                if frame_count % OCR_FREQ == 0:
                    # Center region for OCR
                    # Reduce radius to avoid capturing the ring
                    center_r = int(r * 0.5)
                    center_roi = roi[roi_cy-center_r:roi_cy+center_r, roi_cx-center_r:roi_cx+center_r]
                    
                    # Preprocess for OCR
                    gray_center = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Threshold to get white text as white (255)
                    _, thresh_temp = cv2.threshold(gray_center, 150, 255, cv2.THRESH_BINARY)
                    
                    # Find contours to isolate the letter
                    contours, _ = cv2.findContours(thresh_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Find largest contour (the letter)
                        c = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)
                        
                        # Crop to the letter with some padding
                        pad_c = 5
                        x_c = max(0, x - pad_c)
                        y_c = max(0, y - pad_c)
                        w_c = min(thresh_temp.shape[1] - x_c, w + 2*pad_c)
                        h_c = min(thresh_temp.shape[0] - y_c, h + 2*pad_c)
                        
                        letter_crop = thresh_temp[y_c:y_c+h_c, x_c:x_c+w_c]
                        
                        # Invert for Tesseract (Black text on White bg)
                        thresh_center = cv2.bitwise_not(letter_crop)
                        
                        # Resize to a standard height (e.g., 60px) to help Tesseract
                        target_h = 60
                        scale = target_h / h_c
                        target_w = int(w_c * scale)
                        thresh_center = cv2.resize(thresh_center, (target_w, target_h))
                        
                        # Add a white border
                        thresh_center = cv2.copyMakeBorder(thresh_center, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                        
                        if debug_mode:
                            cv2.imshow("OCR Input", thresh_center)
                        
                        # Try different PSM modes
                        # 10: Single char, 7: Single line, 6: Block
                        psm_modes = [10]
                        found_text = None
                        
                        for psm in psm_modes:
                            config = f"--psm {psm} -c tessedit_char_whitelist=WASDQER"
                            try:
                                text = pytesseract.image_to_string(thresh_center, config=config).strip()
                                print(f"PSM {psm}: '{text}'") # Debug
                                if text and text in ['W', 'A', 'S', 'D', 'Q', 'E', 'R']:
                                    found_text = text
                                    break
                            except Exception as e:
                                print("OCR Error: ", e)
                                pass
                        
                        if found_text:
                            cached_letter = found_text
                        else:
                             # Fallback: Try without whitelist to see what it sees
                            try:
                                raw_text = pytesseract.image_to_string(thresh_center, config="--psm 10").strip()
                                print(f"Raw OCR (No Whitelist): '{raw_text}'")
                            except:
                                pass
    
                if debug_mode:
                    cv2.putText(debug_img, f"Letter: {cached_letter}", (cx - r, cy - r - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
                # 3. Detect Red Line & White Bar (Fast - Run every frame)
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Define the ring area (Annulus)
                # The detected circle 'r' seems to be the inner edge of the track.
                # So we look from r*0.9 to r*1.3
                mask_ring_area = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.circle(mask_ring_area, (roi_cx, roi_cy), int(r * 1.3), 255, -1)
                cv2.circle(mask_ring_area, (roi_cx, roi_cy), int(r * 0.9), 0, -1)
                
                # Red color range
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 100, 100])
                upper_red2 = np.array([180, 255, 255])
                
                mask_red_raw = cv2.inRange(hsv_roi, lower_red1, upper_red1) + cv2.inRange(hsv_roi, lower_red2, upper_red2)
                mask_red = cv2.bitwise_and(mask_red_raw, mask_red_raw, mask=mask_ring_area)
                
                # Find red contours
                contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                red_angle = None
                if contours_red:
                    # Assume largest red blob is the cursor
                    c = max(contours_red, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        rx = int(M["m10"] / M["m00"])
                        ry = int(M["m01"] / M["m00"])
                        red_angle = get_angle((roi_cx, roi_cy), (rx, ry))
                        
                        if debug_mode:
                            # Draw red blob center on debug
                            cv2.circle(debug_img, (x1 + rx, y1 + ry), 5, (0, 0, 255), -1)
    
                # White color range
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 50, 255])
                mask_white_raw = cv2.inRange(hsv_roi, lower_white, upper_white)
                
                mask_white = cv2.bitwise_and(mask_white_raw, mask_white_raw, mask=mask_ring_area)
                
                # Morphological opening
                kernel = np.ones((3,3), np.uint8)
                mask_white_cleaned = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)
                
                contours_white, _ = cv2.findContours(mask_white_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                target_found = False
                if contours_white and red_angle is not None:
                    for wc in contours_white:
                        if cv2.contourArea(wc) < 50: 
                            continue
                        
                        # Check if the red cursor's center is close to this white blob
                        dist = cv2.pointPolygonTest(wc, (rx, ry), True)
                        
                        if dist > -10: # Tolerance
                            target_found = True
                            if debug_mode:
                                cv2.drawContours(debug_img, [wc + (x1, y1)], -1, (255, 0, 0), 2)
                            break
                
                if target_found and cached_letter:
                    if debug_mode:
                        cv2.putText(debug_img, "MATCH!", (cx, cy + r + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if not simulation_mode:
                        time.sleep(0.2)
                        pydirectinput.press(cached_letter.lower())
                        # time.sleep(0.1) # Debounce
            
        # Show debug windows
        # Resize for better visibility if needed
        if debug_mode:
            cv2.imshow('Fishing Debug', debug_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            simulation_mode = not simulation_mode
            print(f"Simulation Mode: {simulation_mode}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug Mode: {debug_mode}")
            if not debug_mode:
                cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
